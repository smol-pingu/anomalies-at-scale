"""Fold Exathlon's two ground-truth intervals into the single label the pipeline scores against.

Exathlon does not ship one interval per anomaly; it ships two nested ones. ``root_cause_start``
to ``root_cause_end`` is when the disturbance was actually applied, and ``extended_effect_end``
is a best-effort estimate of when the metrics returned to normal afterwards. This module unions
them - ``root_cause_start`` through ``extended_effect_end`` - into one interval per anomaly, and
that union is what the paper's AD levels score against.

The fallback matters: ``extended_effect_end`` is blank for some instances, and there the label
is the root-cause interval alone rather than an interval that ends before it starts.

From timestamps to indices
--------------------------
The ground truth is in Unix seconds and everything downstream counts in points, so each
timestamp has to be *located* in its own trace's ``t`` column rather than subtracted from the
first one. Every trace is missing a handful of seconds - between 7 and 22 of them - so
``(timestamp - t[0])`` would be off by the number of gaps preceding it, and off by a different
amount in every trace. Hence :func:`numpy.searchsorted` against the real time column.

What is written
---------------
``anomalous`` intervals rather than per-point labels, which is one of the two shapes
:mod:`~anomalies_scale.stream_evaluation` accepts. Intervals are exact and small; a per-point
array over 2.3 million points would be neither.

The optional second output keeps one row per anomaly instance with its ``anomaly_type``. That
is deliberately not folded into the interval form: overlapping instances of different types
merge into one interval, so the type survives only if it is written separately, and per-type
recall is the most interesting breakdown this dataset offers.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from anomalies_scale.canonical_streams import STREAM_COLUMN

#: Column `stream_evaluation` reads interval-shaped ground truth from. Defined here rather than
#: imported, because it lives in `stream_scoring` and importing that would pull FAISS in to
#: build a label file.
ANOMALOUS_COLUMN = "anomalous"

#: The time column of every raw Exathlon trace, in Unix seconds at 1 Hz.
TIME_COLUMN = "t"

#: Ground-truth columns this reads, in the order the published table declares them.
TRUTH_COLUMNS = ("trace_name", "trace_type", "anomaly_type",
                 "root_cause_start", "root_cause_end", "extended_effect_end")


def side_of(name):
    """``'corpus'`` or ``'test'``, from the trace name alone.

    Trace names are ``<app_id>_<type_id>_<input_rate>_<trace_id>``, and ``type_id`` 0 marks an
    undisturbed run. Reading it from the name rather than from which directory the file sat in
    keeps the per-instance table self-contained.
    """
    parts = str(name).split("_")
    return "corpus" if len(parts) > 1 and parts[1] == "0" else "test"


def read_ground_truth(path):
    """Read ``ground_truth.csv`` and add the unified interval as ``start`` / ``end``.

    ``end`` is the later of ``extended_effect_end`` and ``root_cause_end``, so an extended
    effect that was recorded as ending before the disturbance stopped - or not recorded at all -
    still yields an interval that contains the root cause.
    """
    truth = pd.read_csv(path)
    missing = [c for c in TRUTH_COLUMNS if c not in truth.columns]
    if missing:
        raise ValueError(
            "{0} is not the Exathlon ground truth - it is missing {1}. Its columns are "
            "{2}".format(path, missing, list(truth.columns)))

    truth[TRUTH_COLUMNS[0]] = truth[TRUTH_COLUMNS[0]].astype(str).str.strip()
    effect_end = truth["extended_effect_end"].fillna(truth["root_cause_end"])
    return truth.assign(start=truth["root_cause_start"],
                        end=np.maximum(effect_end, truth["root_cause_end"]))


def merge_intervals(intervals):
    """Merge overlapping or adjacent ``[lo, hi]`` pairs.

    Several traces carry anomalies close enough together that their extended effects run into
    one another. Left unmerged, the overlap is counted twice by anything that sums interval
    lengths, and a point inside both would be covered by two ranges at once.
    """
    merged = []
    for lo, hi in sorted(intervals):
        if merged and lo <= merged[-1][1] + 1:
            merged[-1][1] = max(merged[-1][1], hi)
        else:
            merged.append([lo, hi])
    return merged


def trace_files(traces):
    """Resolve `traces` - directories, files, or a mix - to a sorted list of CSV paths."""
    resolved = []
    for entry in [Path(t) for t in ([traces] if isinstance(traces, (str, Path)) else traces)]:
        resolved.extend(sorted(entry.glob("*.csv")) if entry.is_dir() else [entry])
    return resolved


def locate(time, start, end):
    """Index range of one anomaly inside one trace's time column, or None if it falls outside.

    Half-open on neither side: `lo` is the first sample at or after `start`, `hi` the last at
    or before `end`, so the pair is inclusive at both ends like the rest of this project's
    intervals.
    """
    lo = int(np.searchsorted(time, start, side="left"))
    hi = int(np.searchsorted(time, end, side="right")) - 1
    if lo >= len(time) or hi < lo:
        return None
    return lo, min(hi, len(time) - 1)


def labels_from_ground_truth(traces, ground_truth, output_path=None, detail_path=None,
                             show_progress=False):
    """Per-trace anomalous intervals for a set of raw Exathlon trace CSVs.

    Parameters
    ----------
    traces : path or iterable of paths
        Trace CSVs, or directories of them. Undisturbed traces are expected here too: they
        carry no anomalies, and a row with an empty interval list says that positively, where
        an absent row would only say the trace was never looked at.
    ground_truth : path
        The published ``ground_truth.csv``.
    output_path : path, optional
        Where to write the ``stream`` / ``anomalous`` parquet.
    detail_path : path, optional
        Where to write the per-instance CSV, which keeps ``anomaly_type``.

    Returns
    -------
    pd.DataFrame
        ``stream`` and ``anomalous``, one row per trace.
    """
    truth = read_ground_truth(ground_truth)
    by_trace = dict(list(truth.groupby("trace_name")))

    rows, instances, unplaced = [], [], 0
    for path in trace_files(traces):
        name = path.stem
        # Only the time column is read; the other 2,283 are irrelevant here and reading them
        # would turn a label build into a 25 GB pass.
        time = pd.read_csv(path, usecols=[TIME_COLUMN])[TIME_COLUMN].to_numpy()

        pairs = []
        for record in by_trace.get(name, pd.DataFrame(columns=truth.columns)).itertuples(
                index=False):
            placed = locate(time, record.start, record.end)
            if placed is None:
                unplaced += 1
                continue
            pairs.append(list(placed))
            instances.append({
                "trace": name,
                "side": side_of(name),
                "trace_type": record.trace_type,
                "anomaly_type": record.anomaly_type,
                "root_cause_start": record.root_cause_start,
                "root_cause_end": record.root_cause_end,
                "extended_effect_end": record.extended_effect_end,
                "index_start": placed[0],
                "index_end": placed[1],
                "points": placed[1] - placed[0] + 1,
                "length": len(time),
            })

        rows.append({STREAM_COLUMN: name, ANOMALOUS_COLUMN: merge_intervals(pairs),
                     "length": len(time)})

        if show_progress:
            print("{0:<20} {1:>7,} point(s), {2} anomal{3}".format(
                name, len(time), len(pairs), "y" if len(pairs) == 1 else "ies"))

    labels = pd.DataFrame(rows, columns=[STREAM_COLUMN, ANOMALOUS_COLUMN, "length"])
    detail = pd.DataFrame(instances)

    named = set(truth["trace_name"])
    absent = sorted(named - set(labels[STREAM_COLUMN]))
    if absent:
        print("warning: ground truth names {0} trace(s) with no CSV here: {1}".format(
            len(absent), absent[:6]))
    if unplaced:
        print("warning: {0} anomal{1} fell outside the trace it names".format(
            unplaced, "y" if unplaced == 1 else "ies"))

    if show_progress and len(labels):
        anomalous = labels[ANOMALOUS_COLUMN].map(
            lambda pairs: sum(hi - lo + 1 for lo, hi in pairs))
        points = int(labels["length"].sum())
        print("\n{0} trace(s), {1:,} point(s); {2:,} anomalous ({3:.2%}) across {4} instance(s), "
              "{5}/{0} trace(s) carry one".format(
                  len(labels), points, int(anomalous.sum()),
                  anomalous.sum() / max(points, 1), len(detail),
                  int((labels[ANOMALOUS_COLUMN].map(len) > 0).sum())))

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        storable = labels[[STREAM_COLUMN, ANOMALOUS_COLUMN]].copy()
        storable[ANOMALOUS_COLUMN] = [[[int(lo), int(hi)] for lo, hi in pairs]
                                      for pairs in storable[ANOMALOUS_COLUMN]]
        storable.to_parquet(output_path, index=False)
        if show_progress:
            print("wrote {0}".format(output_path))

    if detail_path is not None and len(detail):
        detail_path = Path(detail_path)
        detail_path.parent.mkdir(parents=True, exist_ok=True)
        detail.to_csv(detail_path, index=False)
        if show_progress:
            print("wrote {0}".format(detail_path))

    return labels[[STREAM_COLUMN, ANOMALOUS_COLUMN]]
