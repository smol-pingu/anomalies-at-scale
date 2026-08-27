"""Detect anomalous intervals by rebuilding the model of normality at every interval.

The pooled detector builds one corpus, one covariance and one index, then searches every
stream against them. This one inverts that: at each interval it restricts the corpus of
normality to that interval, fits a covariance to it, and calibrates a threshold from it.

Nothing is shared between intervals, so an interval is judged only against normal data of
exactly its own width and position. That is the point of the method, and the reason it has no
index, no persisted corpus and no calibration stage: those exist, but one per interval, each
alive only while that interval is being asked about.

The search is the same one the pooled workflow runs
---------------------------------------------------
`widest_first_segment`, unchanged: find the widest clean block, push its boundaries outward in
steps of ``2 ** tol`` for as long as they keep passing, record it, and recurse into the regions
either side. The anomalous regions are the complement of what was found clean.

Only the characteristic function differs between the two detectors - what "clean" means, not
how the stream is swept. That is deliberate: it makes the two directly comparable, and it is
why boundary extension is here rather than a plain dyadic bisection. Without it a normal
section could only ever end on a dyadic boundary.

What it costs
-------------
The search runs stream by stream, but the models it needs are cached across all of them, so
each distinct interval is built once for the whole run however many streams ask about it.
That matters most for extension, which asks about intervals that are not tree nodes and where
different streams frequently extend to the same bounds.

Extension is nonetheless the expensive part here in a way it is not for the pooled detector.
There, an extra interval is one more query against a prebuilt index; here it is a fresh corpus
slice, covariance fit and threshold - a whole model. Smaller ``tol`` localises better and
costs proportionally more.

Per interval
------------
1. Sign the whole corpus of normality, restricted to this interval.
2. Fit ``Sigma^-1/2`` to those signatures - all of them - rank-truncated by retained variance.
3. Split the corpus 95/5. Index the 95%; score the withheld 5% against it; take the statistic
   of those distances as this interval's threshold.
4. Score the stream against the same 95% index the threshold was measured on.

Scoring against the same subset the threshold came from is deliberate. Calibrating on one
reference set and testing against a larger one biases every comparison, because adding points
can only reduce nearest-neighbour distances - measured elsewhere in this project at between
1.6x and 4.7x. This is also the recipe the toy notebook's own per-interval detector settled on.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import faiss
import iisignature

from anomalies_scale.anomaly_detection_pooled import invert_clean, widest_first_segment
from anomalies_scale.canonical_streams import iter_streams
from anomalies_scale.signature_computer import add_base_point, write_corpus

#: Columns of the returned table - the same schema, and the same reporting convention, as the
#: pooled workflow's scoring stage. The two detectors disagree about how to model normality,
#: not about what to report, so one evaluation stage serves both and their outputs can be
#: compared directly.
INTERVAL_COLUMNS = (
    "stream", "start_index", "end_index", "n_points", "start_time", "end_time",
    "depth", "score", "threshold", "ratio", "exceeds_threshold", "tested_in_search",
)


def load_stream_set(files, time_col=None, value_cols=None, stream_col=None, label=""):
    """Stack every stream in `files` into one ``(n, length, width)`` array.

    Every stream must be the same length. That is not a convenience: the search slices the
    normal and test sets to a common ``[lo, hi]`` at each step, so an index means the same
    thing in every stream or it means nothing.
    """
    names, paths = [], []
    for file in files:
        for name, path in iter_streams(file, stream_col=stream_col, time_col=time_col,
                                       value_cols=value_cols):
            names.append(name)
            paths.append(path)

    if not paths:
        raise ValueError("no {0}streams found in {1} file(s)".format(
            label + " " if label else "", len(list(files))))

    lengths = {path.shape[0] for path in paths}
    if len(lengths) > 1:
        raise ValueError(
            "{0}streams have differing lengths {1}; the search slices them all to the same "
            "intervals, so they must share one index space".format(
                label + " " if label else "", sorted(lengths)))

    return names, np.stack(paths)


def sign_slice(block, trunc):
    """Base-pointed signatures of an already-sliced ``(n, points, channels)`` block.

    Split out from :func:`sign_interval` because a block may have been reduced to latent
    channels between the slicing and the signing, in which case its width is no longer the
    width of the streams it came from - and the base point has to be taken from the block that
    is actually signed, not from the original.
    """
    raw = iisignature.sig(block, trunc)
    return add_base_point(raw, block[:, 0, :], block.shape[2], trunc)


def sign_interval(paths, lo, hi, trunc):
    """Base-pointed signatures of every stream in `paths`, restricted to points ``lo..hi``.

    Signed in one batched call rather than stream by stream, which is the whole reason the
    streams are held as one ``(n, length, width)`` array: every stream shares an index space,
    so restricting them all to an interval is a slice.
    """
    return sign_slice(paths[:, lo:hi + 1, :], trunc)


class IntervalModel:
    """The model of normality at one interval: a metric, a reference set, and a threshold.

    Whether the channels are raw or latent is not this class's business. Any channel reduction
    happens once, upstream, before either corpus or test stream reaches the detector, so both
    sides of every comparison are already in the same space by construction.
    """

    def __init__(self, projection, index, threshold, rank, n_train, n_withheld):
        self.projection = projection
        self.index = index
        self.threshold = threshold
        self.rank = rank
        self.n_train = n_train
        self.n_withheld = n_withheld

    def score(self, signatures):
        """Squared Mahalanobis distance from each signature to its nearest normal neighbour."""
        queries = np.ascontiguousarray(
            np.atleast_2d(np.asarray(signatures, dtype=np.float32)) @ self.projection)
        distances, _ = self.index.search(queries, 1)
        return distances[:, 0]


def build_interval_model(corpus_signatures, statistic=np.max, variance_keep=0.999,
                         withhold_fraction=0.05, random_state=0):
    """Fit the metric and calibrate the threshold for one interval.

    The covariance is fitted on the entire corpus of normality at this interval. The threshold
    then comes from a train/withhold split: the withheld streams are scored against the final
    reference set - which contains them - with each one's own zero-distance self-match skipped,
    so the threshold is on the same scale as the distances scoring will produce.

    Both halves matter. Measuring on withheld data keeps the threshold off the corpus's own
    in-sample distances, which understate what unseen normal data scores. Measuring against the
    *final* index keeps it off a reference set sparser than the real one, which would overstate
    them. The two biases run in opposite directions and each is worth several-fold.
    """
    n_streams = corpus_signatures.shape[0]
    if n_streams < 3:
        raise ValueError(
            "need at least 3 normal streams to split and calibrate an interval, got {0}"
            .format(n_streams))

    projection, rank = fit_projection(corpus_signatures, variance_keep)

    n_withheld = max(1, int(round(withhold_fraction * n_streams)))
    order = np.random.default_rng(random_state).permutation(n_streams)
    withheld, train = order[:n_withheld], order[n_withheld:]

    whitened = np.ascontiguousarray(
        corpus_signatures.astype(np.float32) @ projection)

    # The reference set is the whole corpus at this interval - train *and* withheld. Coverage
    # of normality is what a nearest-neighbour detector is short of, and a withheld stream is
    # normal data; excluding it from the reference to keep it pure for calibration would throw
    # away exactly the thing the search is looking for.
    index = faiss.IndexFlatL2(projection.shape[1])
    index.add(whitened)

    # Which forces the calibration to exclude self-matches. Each withheld row is now in the
    # index, so its own zero-distance match comes back first and the second neighbour is the
    # one that means anything - the k=2 trick.
    #
    # The alternative - calibrate against a train-only index, then add the withheld rows
    # afterwards - measures the threshold against a sparser reference than the one scoring
    # will search. Every distance is larger there, so the threshold comes out too high and
    # the detector under-flags. That is the "naive threshold" the Omni notebook's section 5.3
    # measures directly, and the reason the order here is build-then-calibrate rather than
    # calibrate-then-add.
    distances, _ = index.search(np.ascontiguousarray(whitened[withheld]), 2)
    threshold = float(statistic(distances[:, 1]))

    return IntervalModel(projection, index, threshold, rank, len(train), n_withheld)


def scale_depth(lo, hi, span):
    """Dyadic depth an interval's width corresponds to, on a log scale.

    Reported intervals are complements of clean blocks and need not be tree nodes at all, so
    their depth is answered by width rather than by tree position.
    """
    width = max(hi - lo, 1) / max(span, 1)
    return int(round(-np.log2(min(max(width, 1e-12), 1.0))))


def detect_anomalous_intervals(normal_paths, test_paths, test_names, trunc,
                               sig_tol=2, tol=0, variance_keep=0.999, statistic=np.max,
                               withhold_fraction=0.05, random_state=0,
                               show_progress=False):
    """Segment every test stream into normal and anomalous sections.

    Parameters
    ----------
    normal_paths : np.ndarray
        ``(n_normal, length, width)``, time in the first channel. The corpus of normality.
    test_paths : np.ndarray
        ``(n_test, length, width)`` on the same index space - the intervals are shared, so
        the two must agree on length.
    test_names : sequence of str
        Stream identifiers, in the order of `test_paths`.
    sig_tol : int
        Resolution floor as a power of two, in points. The search stops bisecting here, so
        anomalies shorter than ``2 ** sig_tol`` cannot be localised.

    Returns
    -------
    (pd.DataFrame, dict)
        One row per maximal anomalous range per stream, and diagnostics on the search.
    """
    normal_paths = np.asarray(normal_paths, dtype=float)
    test_paths = np.asarray(test_paths, dtype=float)
    if normal_paths.shape[1] != test_paths.shape[1]:
        raise ValueError(
            "normal streams are {0} points and test streams {1}; both are sliced to the same "
            "intervals, so they must share one index space"
            .format(normal_paths.shape[1], test_paths.shape[1]))

    last = test_paths.shape[1] - 1
    minimum = 2 ** sig_tol
    time = test_paths[:, :, 0]

    models = {}
    stats = {"intervals_built": 0, "tests": 0, "model_reuse": 0, "retested": 0,
             "ranks": [], "clean_blocks": 0}

    def model_for(lo, hi):
        """The model of normality at one interval, built once and shared by every stream.

        This is the whole cost of the method, and it does not depend on which stream is
        asking - so although the search runs stream by stream, the models it needs are built
        across all of them. Boundary extension in particular asks about intervals that are
        not tree nodes, and different streams frequently extend to the same bounds.

        Any channel reduction has already happened upstream, so this signs whatever channels
        arrive - raw or latent - and nothing here needs to know which.
        """
        if (lo, hi) not in models:
            models[(lo, hi)] = build_interval_model(
                sign_interval(normal_paths, lo, hi, trunc), statistic=statistic,
                variance_keep=variance_keep, withhold_fraction=withhold_fraction,
                random_state=random_state)
            stats["intervals_built"] += 1
            stats["ranks"].append(models[(lo, hi)].rank)
        else:
            stats["model_reuse"] += 1
        return models[(lo, hi)]

    records = []
    for position, name in enumerate(test_names):
        stream = test_paths[position:position + 1]
        measured = {}

        def measure(lo, hi):
            """Distance from this stream's interval to the corpus at that same interval."""
            lo, hi = max(0, int(lo)), min(last, int(hi))
            if (lo, hi) not in measured:
                model = model_for(lo, hi)
                stats["tests"] += 1
                measured[(lo, hi)] = (
                    float(model.score(sign_interval(stream, lo, hi, trunc))[0]),
                    model.threshold)
            return measured[(lo, hi)]

        def is_clean(lo, hi):
            score, threshold = measure(lo, hi)
            return score <= threshold

        # The same search the pooled workflow runs: find the widest clean block, extend its
        # boundaries outward while they still pass, record it, and recurse into the regions
        # either side. Only the characteristic function differs between the two detectors.
        clean = widest_first_segment(0, last, is_clean, minimum,
                                     floor=2 ** tol if tol else None)
        stats["clean_blocks"] += len(clean)
        searched = set(measured)

        # The anomalous regions are the complement of what was found clean. A complement
        # range is assembled from the gaps between clean blocks, so it need never have been
        # tested as a unit - each is scored directly before being reported.
        for lo, hi in invert_clean(clean, last):
            score, threshold = measure(lo, hi)
            if (lo, hi) not in searched:
                stats["retested"] += 1
            records.append({
                "stream": name,
                "start_index": int(lo), "end_index": int(hi), "n_points": int(hi - lo + 1),
                "start_time": float(time[position, lo]),
                "end_time": float(time[position, hi]),
                "depth": scale_depth(lo, hi, last),
                "score": score, "threshold": threshold,
                "ratio": score / threshold if threshold else float("inf"),
                # A range that fails at the scales the search bisected to can still pass when
                # measured whole; that is what a diffuse departure looks like, and it is
                # reported rather than dropped.
                "exceeds_threshold": bool(score > threshold),
                "tested_in_search": (lo, hi) in searched,
            })

        if show_progress and (position + 1) % 25 == 0:
            print("  {0}/{1} streams, {2} interval model(s) built".format(
                position + 1, len(test_names), stats["intervals_built"]))

    table = pd.DataFrame.from_records(records, columns=list(INTERVAL_COLUMNS))
    table = table.sort_values(["stream", "start_index"]).reset_index(drop=True)

    stats["mean_rank"] = float(np.mean(stats["ranks"])) if stats["ranks"] else float("nan")
    stats["streams_flagged"] = int(table["stream"].nunique()) if len(table) else 0
    stats["streams_clean"] = len(test_names) - stats["streams_flagged"]
    del stats["ranks"]

    if show_progress:
        print("{intervals_built} interval model(s) built and {model_reuse} reused over "
              "{tests} test(s); {streams_flagged} stream(s) flagged, {streams_clean} clean; "
              "mean metric rank {mean_rank:.0f}".format(**stats))
    return table, stats


def parse_statistic(name):
    """Turn a statistic name into the function that reduces withheld distances to a threshold.

    Accepts ``'max'``, ``'median'``, and percentiles written ``'p99'`` or ``'p99.9'``.
    """
    if name == "max":
        return np.max
    if name == "median":
        return np.median
    if isinstance(name, str) and name.startswith("p"):
        try:
            percentile = float(name[1:])
        except ValueError:
            raise ValueError("could not read {0!r} as a percentile".format(name)) from None
        return lambda values: np.percentile(values, percentile)
    raise ValueError("statistic must be 'max', 'median' or a percentile such as 'p99.9', "
                     "got {0!r}".format(name))


def detect(corpus_files, test_files, output_path=None, diagnostics_path=None, trunc=4,
           sig_tol=2, tol=0, variance_keep=0.999, statistic="max", withhold_fraction=0.05,
           random_state=0, time_col=None, value_cols=None, stream_col=None,
           show_progress=False):
    """Stage entry point: load both corpora, segment the test streams, write the intervals.

    Whatever channels the streams arrive in are the channels this signs. Any reduction was
    applied upstream to both sides at once, so nothing here refits, remembers or reverses one.
    """
    import json

    corpus_names, corpus_paths = load_stream_set(
        corpus_files, time_col, value_cols, stream_col, label="normal")
    test_names, test_paths = load_stream_set(
        test_files, time_col, value_cols, stream_col, label="test")

    if show_progress:
        print("{0} normal stream(s), {1} test stream(s), {2} points, {3} channel(s)".format(
            len(corpus_names), len(test_names), test_paths.shape[1],
            test_paths.shape[2] - 1))

    table, stats = detect_anomalous_intervals(
        corpus_paths, test_paths, test_names, trunc=trunc, sig_tol=sig_tol, tol=tol,
        variance_keep=variance_keep, statistic=parse_statistic(statistic),
        withhold_fraction=withhold_fraction, random_state=random_state,
        show_progress=show_progress)

    stats["n_normal_streams"] = len(corpus_names)
    stats["n_test_streams"] = len(test_names)
    stats["statistic"] = statistic

    if output_path is not None:
        write_corpus(table, output_path)
        if show_progress:
            print("wrote {0} anomalous range(s) to {1}".format(len(table), output_path))

    if diagnostics_path is not None:
        from pathlib import Path

        diagnostics_path = Path(diagnostics_path)
        diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
        with open(diagnostics_path, "w", encoding="utf-8") as handle:
            json.dump(stats, handle, indent=2, default=int)

    return table, stats
