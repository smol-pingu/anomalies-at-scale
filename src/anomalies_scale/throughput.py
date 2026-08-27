"""Pair what each stage cost with how much work it did, and derive the rates.

Snakemake's ``benchmark:`` directive records *time* - wall-clock seconds, peak resident memory,
CPU seconds, I/O - one TSV per job. On its own that is not throughput: a stage taking 40 seconds
means nothing until you know whether it built thirty thousand corpus intervals or three million.
This module supplies the denominators and does the division.

Two costs, and they answer different questions
----------------------------------------------
``build``
    canonicalise, preprocess, window, corpus, covariance, index, calibrate. What it costs to
    onboard a fleet. Paid once, and again on every retrain.
``serve``
    score. What it costs to watch data arriving. Paid forever.

A pipeline can be comfortable on one and hopeless on the other, and reporting them together
hides exactly that. The ratio between them - ``build_per_score_second`` - is the break-even: how
much data must be scored before the one-off build stops dominating, and therefore how often
retraining is a sane thing to do.

Missing pieces are skipped rather than raised on. A run made before benchmarking existed, or one
where a stage was cached and so never re-timed, should still produce a summary.
"""

from __future__ import annotations

import json
from pathlib import Path

#: Columns Snakemake writes into a benchmark TSV. Which are present varies with version, so
#: every one of them is optional and read by name rather than by position.
BENCHMARK_COLUMNS = ("s", "max_rss", "max_vms", "max_uss", "max_pss",
                     "io_in", "io_out", "mean_load", "cpu_time")

#: Stages whose cost is paid once per corpus, against the one paid per batch of new data.
BUILD_STAGES = ("canonicalise", "umap", "preprocess", "window", "corpus", "normalise",
                "reduce", "covariance", "index", "forest", "calibrate")
SERVE_STAGES = ("score",)


#: The rules whose benchmark files are collected into the summary. `evaluate` carries a
#: `benchmark:` of its own but is deliberately absent here: Snakemake writes a rule's TSV when
#: the rule *finishes*, so the stage that assembles the summary cannot observe its own cost, and
#: reading the file anyway would quietly report the previous run's.
STAGE_NAMES = ("canonicalise", "umap", "preprocess", "window", "corpus", "normalise",
               "reduce", "covariance", "index", "forest", "calibrate", "score")


def peak_memory_bytes():
    """High-water mark of this process's resident memory, or ``None`` where unavailable.

    Worth measuring in-process because Snakemake's own `benchmark:` cannot do it here: these
    rules are `run:` directives executing inside the workflow process, so on Windows there is no
    child for the benchmarker's poller to watch and every memory column comes back ``NA``. Wall
    clock still lands, so the TSVs remain the source for time and this covers memory.
    """
    try:
        import psutil

        info = psutil.Process().memory_info()
        # Windows exposes the true peak; elsewhere `rss` is the current value and the resource
        # module below is the better answer, so it is only a fallback.
        if hasattr(info, "peak_wset"):
            return int(info.peak_wset)
    except Exception:                              # noqa: BLE001 - diagnostics must never fail
        pass

    try:
        import resource

        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # Kilobytes on Linux, bytes on macOS.
        return int(peak if peak > 1 << 32 else peak * 1024)
    except Exception:                              # noqa: BLE001
        return None


def discover_benchmarks(directory, dataset):
    """Group every benchmark TSV belonging to one dataset under its stage name.

    A rule with wildcards writes one file per job - `canonicalise_corpus` and
    `canonicalise_test`, `window_corpus_w16` and `window_test_w16` - so a stage maps to a list
    rather than a single path. Longest stage name first, so that `corpus` does not claim
    `window_corpus_w16`.
    """
    directory = Path(directory)
    if not directory.exists():
        return {}

    found = {}
    for path in sorted(directory.glob("{0}_*.tsv".format(dataset))):
        suffix = path.stem[len(dataset) + 1:]
        for stage in sorted(STAGE_NAMES, key=len, reverse=True):
            if suffix == stage or suffix.startswith(stage + "_"):
                found.setdefault(stage, []).append(path)
                break
    return found


def read_benchmark(paths):
    """One stage's benchmark TSVs folded into a single dict, or ``None`` if there are none.

    Snakemake writes a header row and one row per job. Where a stage ran several jobs the
    numbers are summed for wall-clock, CPU and I/O but *maximised* for memory: peak RSS across
    jobs is a high-water mark, not a total, and adding two peaks would describe a machine
    nobody ran on.
    """
    if isinstance(paths, (str, Path)):
        paths = [paths]

    rows = []
    for path in paths:
        path = Path(path)
        if not path.exists():
            continue
        lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if len(lines) < 2:
            continue

        header = lines[0].split("\t")
        for line in lines[1:]:
            cells = line.split("\t")
            row = {}
            for name, cell in zip(header, cells):
                if name not in BENCHMARK_COLUMNS:
                    continue
                try:
                    row[name] = float(cell)
                except ValueError:                 # '-' where a metric was unavailable
                    continue
            rows.append(row)

    if not rows:
        return None

    combined = {"n_jobs": len(rows)}
    for name in BENCHMARK_COLUMNS:
        values = [row[name] for row in rows if name in row]
        if not values:
            continue
        combined[name] = max(values) if name.startswith("max_") else sum(values)

    if combined.get("s") and combined.get("cpu_time"):
        # How many cores the stage actually used. Near 1 means single-threaded and there is
        # headroom; near the core count means the machine was saturated and these timings will
        # not transfer to a smaller one. Omitted rather than reported as zero where Snakemake
        # could not measure CPU time - which is every `run:` rule on Windows.
        combined["cores_used"] = round(combined["cpu_time"] / combined["s"], 2)
    return combined


def read_json(path):
    path = Path(path) if path else None
    if path is None or not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return {}


def file_bytes(path):
    """Size on disk, following a directory to the sum of what it holds."""
    path = Path(path) if path else None
    if path is None or not path.exists():
        return None
    if path.is_dir():
        return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    return path.stat().st_size


def collect(benchmarks=None, artifacts=None, score_stats=None, covariance_diagnostics=None):
    """Assemble counts, sizes, per-stage costs and the rates that follow from them.

    Parameters
    ----------
    benchmarks : mapping of stage name to path or list of paths
        The TSVs Snakemake's ``benchmark:`` wrote, as :func:`discover_benchmarks` returns them.
    artifacts : mapping of name to path
        Files whose size is worth recording - the corpus, the metric, the index, the scores.
    score_stats : path
        JSON written by :func:`anomalies_scale.stream_scoring.score_streams`.
    covariance_diagnostics : path
        JSON written by :func:`anomalies_scale.covariance_creation.create_covariance`, which
        already carries the corpus interval count, the dimension and the retained rank.
    """
    stages = {}
    for stage, path in (benchmarks or {}).items():
        measured = read_benchmark(path)
        if measured is not None:
            stages[stage] = measured

    sizes = {}
    for name, path in (artifacts or {}).items():
        size = file_bytes(path)
        if size is not None:
            sizes[name] = int(size)

    scored = read_json(score_stats)
    metric = read_json(covariance_diagnostics)

    counts = {}
    for key, value in (("corpus_intervals", metric.get("n_intervals")),
                       ("corpus_streams", metric.get("n_streams")),
                       ("signature_terms", metric.get("dimension")),
                       ("retained_rank", metric.get("rank")),
                       ("test_streams", scored.get("n_streams")),
                       ("points_scored", scored.get("n_points")),
                       ("interval_queries", scored.get("n_queries")),
                       ("reference_vectors", scored.get("reference_size"))):
        if value is not None:
            counts[key] = int(value)

    build_seconds = sum(stages[s].get("s", 0.0) for s in BUILD_STAGES if s in stages)
    serve_seconds = sum(stages[s].get("s", 0.0) for s in SERVE_STAGES if s in stages)

    def per_second(count, seconds):
        if not count or not seconds:
            return None
        return round(count / seconds, 2)

    rates = {
        "build_seconds": round(build_seconds, 3) or None,
        "score_seconds": round(serve_seconds, 3) or None,
        # The break-even: seconds of one-off build per second of scoring. Large means the corpus
        # is expensive to prepare relative to using it, and retraining often is a bad trade.
        "build_per_score_second": (round(build_seconds / serve_seconds, 2)
                                   if build_seconds and serve_seconds else None),
        "corpus_intervals_per_second": per_second(
            counts.get("corpus_intervals"), stages.get("corpus", {}).get("s")),
        "queries_per_second": per_second(counts.get("interval_queries"), serve_seconds),
        # The headline: how much live data one machine can watch.
        "points_scored_per_second": per_second(counts.get("points_scored"), serve_seconds),
        "queries_per_point_scored": (
            round(counts["interval_queries"] / counts["points_scored"], 4)
            if counts.get("interval_queries") and counts.get("points_scored") else None),
        "bytes_per_corpus_interval": (
            round(sizes["index"] / counts["corpus_intervals"], 1)
            if sizes.get("index") and counts.get("corpus_intervals") else None),
    }
    # Peak memory from whichever source has it: Snakemake's poller where the platform supports
    # it, and the scoring pass's own in-process measurement otherwise.
    peak = [stages[s]["max_rss"] for s in stages if stages[s].get("max_rss")]
    if scored.get("peak_rss_bytes"):
        peak.append(scored["peak_rss_bytes"] / 1024 ** 2)
    if peak:
        rates["peak_rss_mb"] = round(max(peak), 1)

    return {"counts": counts, "bytes": sizes, "stages": stages,
            "rates": {key: value for key, value in rates.items() if value is not None}}
