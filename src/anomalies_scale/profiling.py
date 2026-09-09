"""Per-stage wall time and memory, for pipelines whose stages are buried in library calls.

The throughput module answers "what did this run cost" from Snakemake's benchmark TSVs. This
answers the finer question the notebooks need: *which stage* spent the time and held the
memory - the UMAP fit, the corpus signatures, the whitening, the index, the scoring - when
those stages are not separate Snakemake rules but nested function calls inside something like
`bagging.one_draw`.

Two pieces:

`MemoryTrace`
    One background thread sampling this process's resident set at a fixed interval, keeping
    ``(timestamp, rss)``. One sampler for a whole run rather than one per call, because a
    thread per call would cost more than the calls it measures.

`StageRecorder`
    Names a window of time. Given the trace, a window yields the peak RSS *observed while it
    was open*, which is the number that matters for "how much memory did the UMAP use".

Why sampling and not `peak_wset`. Windows' `peak_wset` is a process high-water mark that never
falls, so every stage after the largest one inherits its number and the breakdown is a
constant column. Sampling gives each window its own maximum. The cost is that a spike shorter
than the sample interval can be missed, so the interval is a parameter and the default is
deliberately fine.

Attribution when stages nest. A stage records its own window regardless of what is open around
it, so an outer stage's peak includes its children's. `flatten` therefore reports `depth`, and
`exclusive_seconds` subtracts time spent inside nested stages; memory is left inclusive,
because resident memory genuinely is shared with whatever else is live at that moment and
pretending otherwise would invent a number.
"""

from __future__ import annotations

import functools
import threading
import time
from contextlib import contextmanager


def _resident_bytes():
    """This process's current resident set, or ``None`` when psutil is unavailable."""
    try:
        import psutil

        return int(psutil.Process().memory_info().rss)
    except Exception:                                # noqa: BLE001 - diagnostics never raise
        return None


class MemoryTrace:
    """Samples resident memory on a background thread for as long as it is running."""

    def __init__(self, interval=0.1):
        self.interval = float(interval)
        self.samples = []
        self._stamps = []
        self._stop = None
        self._thread = None

    def _run(self):
        while not self._stop.wait(self.interval):
            value = _resident_bytes()
            if value is None:
                return
            self.samples.append((time.perf_counter(), value))

    def start(self):
        if self._thread is not None:
            return self
        value = _resident_bytes()
        self.samples.append((time.perf_counter(), value or 0))
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def stop(self):
        if self._thread is None:
            return
        self._stop.set()
        self._thread.join(timeout=2.0)
        self._thread = None

    def peak_between(self, start, end):
        """Largest sample in ``[start, end]``, or ``None`` if the window caught none.

        A window shorter than the sample interval can fall between two samples. That returns
        ``None`` rather than a neighbouring sample, so a missing measurement is visible as
        missing instead of being quietly filled in with a wrong one.
        """
        # The samples are appended in time order, so the window is a contiguous slice and
        # bisect finds it in log time. A linear scan here is quadratic over a run: samples
        # accumulate at a fixed rate while the number of stages also grows, and a long
        # experiment ends up spending real time measuring itself.
        import bisect

        stamps = self._stamps
        if len(stamps) != len(self.samples):
            stamps = self._stamps = [stamp for stamp, _ in self.samples]
        lo = bisect.bisect_left(stamps, start)
        hi = bisect.bisect_right(stamps, end)
        if lo >= hi:
            return None
        return max(value for _, value in self.samples[lo:hi])

    def __enter__(self):
        return self.start()

    def __exit__(self, *_):
        self.stop()
        return False


class StageRecorder:
    """Names windows of time and pairs each with its peak memory from a `MemoryTrace`."""

    def __init__(self, trace=None, interval=0.1):
        self.trace = trace if trace is not None else MemoryTrace(interval)
        self.records = []
        self._open = []

    def __enter__(self):
        self.trace.start()
        return self

    def __exit__(self, *_):
        self.trace.stop()
        return False

    @contextmanager
    def stage(self, name, **extra):
        """Record one named stage. Reentrant, and safe to nest."""
        depth = len(self._open)
        started, entry_rss = time.perf_counter(), _resident_bytes()
        marker = {"name": name, "depth": depth, "children": 0.0}
        self._open.append(marker)
        try:
            yield marker
        finally:
            self._open.pop()
            finished, exit_rss = time.perf_counter(), _resident_bytes()
            elapsed = finished - started
            for parent in self._open:
                parent["children"] += elapsed

            peak = self.trace.peak_between(started, finished)
            self.records.append({
                "stage": name,
                "depth": depth,
                "seconds": elapsed,
                "exclusive_seconds": max(elapsed - marker["children"], 0.0),
                "peak_rss_mb": None if peak is None else peak / (1024 * 1024),
                "entry_rss_mb": None if entry_rss is None else entry_rss / (1024 * 1024),
                "delta_rss_mb": (None if None in (entry_rss, exit_rss)
                                 else (exit_rss - entry_rss) / (1024 * 1024)),
                **extra,
            })

    def wrap(self, function, name=None):
        """The same function, recording a stage every time it is called."""
        label = name or getattr(function, "__name__", repr(function))

        @functools.wraps(function)
        def recorded(*args, **kwargs):
            with self.stage(label):
                return function(*args, **kwargs)

        recorded.__wrapped_stage__ = label
        return recorded

    @contextmanager
    def instrument(self, targets):
        """Temporarily replace ``module.attribute`` with a recording wrapper.

        `targets` is an iterable of ``(module, attribute)`` or ``(module, attribute, name)``.
        Patching where a stage is *defined* rather than where it is called means a stage is
        captured no matter how deeply the call sits inside library code - which is the point,
        since `one_draw` builds a corpus, a metric and an index without exposing any of them.

        A module that has already imported the name by value keeps its own reference, so
        anything imported as ``from x import y`` has to be patched in the importing module
        too. Missing attributes are skipped rather than raised on, so one refactor upstream
        does not take the whole run down.
        """
        patched = []
        try:
            for target in targets:
                module, attribute = target[0], target[1]
                label = target[2] if len(target) > 2 else attribute
                original = getattr(module, attribute, None)
                if original is None or not callable(original):
                    continue
                patched.append((module, attribute, original))
                setattr(module, attribute, self.wrap(original, label))
            yield self
        finally:
            for module, attribute, original in patched:
                setattr(module, attribute, original)

    def flatten(self, experiment=None):
        """Every recorded stage as a list of dicts, newest last."""
        rows = [dict(record) for record in self.records]
        if experiment is not None:
            for row in rows:
                row.setdefault("experiment", experiment)
        return rows

    def summary(self, experiment=None):
        """One row per stage name: call count, total time, and the largest peak seen.

        Returned as plain dicts so this module stays free of a pandas import; the notebook
        makes the frame.
        """
        totals = {}
        for record in self.records:
            entry = totals.setdefault(record["stage"], {
                "stage": record["stage"], "calls": 0, "seconds": 0.0,
                "exclusive_seconds": 0.0, "peak_rss_mb": None, "delta_rss_mb": 0.0})
            entry["calls"] += 1
            entry["seconds"] += record["seconds"]
            entry["exclusive_seconds"] += record["exclusive_seconds"]
            entry["delta_rss_mb"] += record["delta_rss_mb"] or 0.0
            peak = record["peak_rss_mb"]
            if peak is not None:
                entry["peak_rss_mb"] = (peak if entry["peak_rss_mb"] is None
                                        else max(entry["peak_rss_mb"], peak))

        rows = sorted(totals.values(), key=lambda r: -r["seconds"])
        if experiment is not None:
            for row in rows:
                row["experiment"] = experiment
        return rows
