"""Choose one threshold per depth by k-fold cross-validation over whole streams.

The question a threshold answers is "how far from the corpus does *unseen* normal data sit?",
and the difficulty is that every way of measuring it on data the corpus already contains gives
an answer that is too small.

Why self-exclusion is not enough
--------------------------------
The obvious approach is to score each corpus interval against the index and skip its own
zero-distance self-match. That removes one row and leaves the problem intact. Every stream
contributes `C(cells + 1, 2)` overlapping grid intervals - 36 at granularity 3 - and they are
all in the index. A corpus interval's nearest surviving neighbour is therefore usually another
interval of *the same stream*, covering nearly the same points. What gets measured is how well
a stream matches itself, which is far closer than how well it matches the rest of the corpus,
and the threshold inherits that closeness.

Measured on C-MAPSS, excluding only the exact self understated the threshold by 3.35x, which
was the difference between flagging 13.6% of points and 37%.

What this does instead
----------------------
Partition the *streams* into `k` folds. For each fold, build a reference index from the streams
in the other folds only, and score every interval of the held-out streams against it. Because
no interval of a held-out stream can match anything belonging to its own stream, the distance
measured is genuinely "unseen stream against corpus" - the condition the detector operates
under when it meets a new stream.

Every interval is scored exactly once, so the calibration sample is the whole corpus rather
than the withheld fraction of it. On the C-MAPSS run that is 3,600 intervals instead of 720,
which matters because the previous sample was too small for the requested percentile to mean
anything: `p99.9` of 120 values interpolates between the largest two, so the threshold was
being set by a single interval.

Buckets
-------
Scores are bucketed by dyadic **depth**, which is what the corpus carries, what the index bands
on, and what `stream_scoring.resolve_threshold` looks up. A level-*k* signature term scales
like the increment to the *k*-th power, so wide intervals produce systematically larger vectors
than narrow ones and one threshold for all of them would behave partly as a width filter.
"""

from __future__ import annotations

from pathlib import Path

import faiss
import numpy as np
import pandas as pd

from anomalies_scale.covariance_creation import (
    load_covariance, read_corpus, signature_matrix)
from anomalies_scale.threshold_calibration import parse_statistic

#: Column identifying which stream an interval came from - what the folds are drawn over.
STREAM_COLUMN = "stream"


def stream_folds(streams, k, random_state=0):
    """Partition unique stream names into `k` folds of near-equal size.

    Folds are over **streams**, never over intervals. Splitting intervals would put a stream's
    own overlapping intervals on both sides of the boundary, which is precisely the leak this
    module exists to close.

    Shuffled before partitioning, because stream names often carry an order - machine-1-1
    through machine-3-11, engines 1 to 100 - and contiguous blocks of that order can share
    properties that make one fold systematically easier than another.
    """
    unique = np.unique(np.asarray(streams))
    if len(unique) < 2:
        raise ValueError(
            "cross-validation needs at least 2 streams to hold one out; the corpus has "
            "{0}".format(len(unique)))

    k = int(k)
    if k < 2:
        raise ValueError("k must be at least 2, got {0}".format(k))
    if k > len(unique):
        raise ValueError(
            "k={0} exceeds the {1} stream(s) available; a fold would be empty. Use k <= {1}, "
            "or k={1} for leave-one-stream-out.".format(k, len(unique)))

    shuffled = np.random.default_rng(random_state).permutation(unique)
    return [np.asarray(fold) for fold in np.array_split(shuffled, k)]


def fold_distances(whitened, depths, streams, folds, band=1, neighbours=1,
                   show_progress=False):
    """Distance to the `neighbours`-th nearest normal, with the interval's own fold held out.

    For each fold and each depth present in it, a flat index is built over the intervals of
    the *other* folds that lie within `band` of that depth, and the held-out intervals at that
    depth are queried against it.

    Rebuilt per fold rather than filtering neighbours out of one shared index: filtering would
    need an unbounded `k` to guarantee enough surviving neighbours, and at these sizes a flat
    index is a memcpy.

    `neighbours` must match what `stream_scoring.StreamScorer` will use. The threshold is a
    quantile of these distances, and the k-th neighbour is systematically further away than the
    first, so calibrating at one k and scoring at another compares numbers drawn from different
    distributions - which would show up as a change in the flagged fraction and read as an
    effect of k rather than as the mismatch it is.

    Returns
    -------
    np.ndarray
        One distance per row of `whitened`, in the same order.
    """
    distances = np.full(len(whitened), np.nan)
    dimension = whitened.shape[1]
    neighbours = int(neighbours)

    for number, fold in enumerate(folds, start=1):
        held = np.isin(streams, fold)
        for depth in np.unique(depths[held]):
            queries = np.flatnonzero(held & (depths == depth))
            reference = np.flatnonzero(~held & (np.abs(depths - depth) <= band))
            if reference.size < neighbours:
                raise ValueError(
                    "fold {0} leaves {1} reference interval(s) within band {2} of depth {3}, "
                    "fewer than the {4} neighbour(s) asked for. Lower detect.neighbours or "
                    "calibrate.folds, or widen the band."
                    .format(number, reference.size, band, depth, neighbours))

            index = faiss.IndexFlatL2(dimension)
            index.add(np.ascontiguousarray(whitened[reference]))
            found, _ = index.search(np.ascontiguousarray(whitened[queries]), neighbours)
            distances[queries] = found[:, neighbours - 1]

        if show_progress:
            print("  fold {0}/{1}: {2} stream(s), {3} interval(s) scored".format(
                number, len(folds), len(fold), int(held.sum())))

    missing = int(np.isnan(distances).sum())
    if missing:
        raise ValueError(
            "{0} interval(s) were never scored; every row must belong to exactly one "
            "fold".format(missing))
    return distances


def source_of(name):
    """Strip a windowing suffix, so ``train_FD001__3_7`` groups with ``train_FD001__3_8``.

    Only ever applied when the corpus is known to be windowed - see the `windowed` argument to
    :func:`crossvalidated_thresholds`. Applied blindly it would mangle any stream whose own name
    happens to end in an underscore and digits, which is common enough to be worth not risking.
    """
    text = str(name)
    head, sep, tail = text.rpartition("_")
    return head if sep and tail.isdigit() else text


def crossvalidated_thresholds(corpora, covariance, k=5, statistic="p99", band=1,
                              neighbours=1, normaliser=None, random_state=0, windowed=False,
                              output_path=None, show_progress=False):
    """One threshold per depth, cross-validated over streams.

    Parameters
    ----------
    corpora : path, DataFrame, or mapping of them
        The corpus rows to calibrate over. A mapping - ``{"fit": ..., "withheld": ...}`` - is
        concatenated, which is what the pipeline wants: every indexed interval takes a turn
        held out, so the calibration sample is the whole corpus.
    covariance : path or array
        ``Sigma^-1/2``. The same matrix the index was whitened with; a different one would
        measure distances in a space the detector does not search.
    k : int
        Number of folds. ``k = n_streams`` is leave-one-stream-out. Note this is *not* the
        neighbour count - see `neighbours`.
    statistic : str
        How a bucket's distances become its threshold - ``'p99'``, ``'max'``, ``'median'``.
    neighbours : int
        Which neighbour's distance is the score: 1 is the nearest, n the n-th nearest. Must
        equal what `stream_scoring` will score with; see :func:`fold_distances`.
    normaliser : Normaliser or path, optional
        Applied before whitening when the corpus was normalised, for the same reason: the
        index lives in normalised units.
    windowed : bool
        Whether the corpus was built from windows rather than whole streams. When it was, folds
        are formed over the *source* streams, so every window of one engine is held out
        together. Sibling windows are not independent - same unit, same wear, same calibration
        offsets - and a fold that put one in the reference set while querying another would
        measure how well an engine matches itself, which is the exact leak this module exists
        to close, reintroduced one level down.

    Returns
    -------
    (dict, pd.DataFrame)
        ``{depth: threshold}`` and the table behind it.
    """
    frames = corpora if isinstance(corpora, dict) else {"corpus": corpora}
    frame = pd.concat([read_corpus(part) for part in frames.values()], ignore_index=True)

    if STREAM_COLUMN not in frame.columns or "depth" not in frame.columns:
        raise ValueError(
            "the corpus needs {0!r} and 'depth' columns to be cross-validated by stream; got "
            "{1}".format(STREAM_COLUMN, list(frame.columns)[:8]))

    if normaliser is not None:
        if isinstance(normaliser, (str, Path)):
            from anomalies_scale.normalisation import Normaliser

            normaliser = Normaliser.load(normaliser)
        frame = normaliser.transform(frame)

    signatures, columns = signature_matrix(frame)
    if isinstance(covariance, (str, Path)):
        covariance = load_covariance(covariance, expected_columns=columns)
    covariance = np.asarray(covariance, dtype=float)

    whitened = np.ascontiguousarray(signatures @ covariance.T, dtype="float32")
    depths = frame["depth"].to_numpy(dtype=int)
    streams = frame[STREAM_COLUMN].to_numpy()
    if windowed:
        streams = np.asarray([source_of(name) for name in streams])

    folds = stream_folds(streams, k, random_state)
    if show_progress:
        print("{0:,} interval(s) over {1} {2}, {3} fold(s), band {4}, {5}".format(
            len(frame), len(np.unique(streams)),
            "source stream(s), windows held out together" if windowed else "stream(s)",
            len(folds), band,
            "nearest neighbour" if int(neighbours) == 1
            else "{0}-th nearest neighbour".format(int(neighbours))))

    distances = fold_distances(whitened, depths, streams, folds, band, neighbours,
                               show_progress)
    reduce_to_threshold = parse_statistic(statistic)

    records = []
    for depth in sorted(np.unique(depths)):
        values = distances[depths == depth]
        records.append({
            "depth": int(depth),
            "relative_width": 2.0 ** -int(depth),
            "n_calibration": int(values.size),
            "threshold": float(reduce_to_threshold(values)),
            "median_distance": float(np.median(values)),
            "p90_distance": float(np.percentile(values, 90)),
            "p99_distance": float(np.percentile(values, 99)),
            "max_distance": float(values.max()),
            # How many samples sit above the chosen threshold. One is a warning: the operating
            # point is then a single interval, and the percentile asked for was finer than the
            # sample can express.
            "n_above_threshold": int((values > reduce_to_threshold(values)).sum()),
        })

    table = pd.DataFrame.from_records(records)
    table.attrs["statistic"] = statistic
    table.attrs["k"] = len(folds)
    table.attrs["neighbours"] = int(neighbours)
    thresholds = dict(zip(table["depth"].astype(int), table["threshold"].astype(float)))

    if show_progress:
        print("\n" + table.to_string(index=False))
        thin = table[table["n_above_threshold"] < 3]
        if len(thin):
            print("\nWARNING: depth(s) {0} have fewer than 3 calibration intervals above the "
                  "threshold, so it rests on almost no evidence. Use a coarser statistic than "
                  "{1!r}, or more streams.".format(
                      list(thin["depth"]), statistic))

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(output_path, index=False)
        if show_progress:
            print("wrote {0}".format(output_path))

    return thresholds, table
