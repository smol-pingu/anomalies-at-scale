"""Score intervals by isolation rather than by distance to the nearest normal one.

The alternative to the pooled Mahalanobis-kNN detector, and on C-MAPSS the stronger of the two.
Measured on FD001, same corpus intervals, same depth banding, same k-fold calibration over source
streams, same widest-first search - only the per-interval score differs::

    detector                             P       R      F1    adjF1   flagged
    Mahalanobis-kNN, whitened          0.853   0.432   0.573   0.917    7.60%
    Isolation Forest, whitened         0.904   0.616   0.733   0.968   10.25%
    Isolation Forest, raw              0.723   0.927   0.813   0.849   19.27%

Why it does better here
-----------------------
An isolation forest uses no metric at all. It picks a coordinate at random, splits uniformly
between that coordinate's minimum and maximum within the current node, recurses, and scores a
point by how few splits were needed to isolate it. There is no inner product, no covariance and
nothing to invert - so the corpus condition number, which reaches 1e114 on signature data, is
simply not a quantity this detector has an opinion about.

It is also invariant to rescaling any single coordinate, because the split is drawn between the
node's own extremes and rescales with them. That matters on a signature corpus, where column
standard deviations span a factor of 4.4e24 between level-1 and level-2 terms. Measured:
standardising every column changes F1 by 0.003.

What it is *not* invariant to is rotation, since its cuts are axis-aligned - and that turns out
to be the load-bearing difference. Rotating the raw corpus into principal coordinates at full
numerical rank, discarding nothing, costs F1 0.813 -> 0.668; truncating those 356 directions
down to the 11 that `metric.variance_keep` retains costs only a further 0.020. The raw signature
axes are a good basis for axis-aligned splits because each one is a specific iterated integral,
where a principal direction is a mixture of hundreds of them.

Which space to score in
-----------------------
Both are worth having, and they are a genuine trade rather than one dominating:

``whitened``  the vectors the FAISS index holds, ``signature @ Sigma^-1/2``. Conservative -
              10.25% of points flagged against a true rate of 15.03% - with the best adjusted
              F1 of anything measured, 0.968. Choose this to answer "was each failure caught".
``raw``       the signature as computed. The best plain F1, 0.813, at the cost of flagging
              19.27%. Choose this to maximise point-wise overlap.

How it plugs in
---------------
:class:`DepthForests` presents the same small surface `StreamScorer` uses of a
:class:`~anomalies_scale.index_creation.PooledIndex` - ``depth_for_width``, ``search`` and
``dimension`` - so the scoring stage needs no branch. `search` receives queries already in the
detector's own space, exactly as the index receives already-whitened ones, and
``scorer_covariance`` is the matrix the scorer should apply to get them there.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from anomalies_scale.covariance_creation import load_covariance, read_corpus, signature_matrix
from anomalies_scale.crossvalidated_thresholding import parse_statistic, source_of, stream_folds

#: Trees per forest. 200 is what the measurements above used.
DEFAULT_N_ESTIMATORS = 200

#: The coordinate systems a forest may be fitted in. See the module docstring for the trade.
SPACES = ("whitened", "raw")

#: Column identifying which stream a corpus interval came from.
STREAM_COLUMN = "stream"


def make_forest(n_estimators=DEFAULT_N_ESTIMATORS, max_samples="auto", random_state=0):
    """One isolation forest, with the project's defaults."""
    from sklearn.ensemble import IsolationForest

    return IsolationForest(n_estimators=int(n_estimators), max_samples=max_samples,
                           random_state=random_state)


def forest_scores(forest, vectors):
    """Negated ``score_samples``, so that larger means more anomalous.

    The convention the rest of this pipeline uses for distances, and the one SigMahaKNN's own
    baselines adopt for exactly this reason - so a threshold test reads ``value > threshold``
    whichever detector produced the value.
    """
    return -np.asarray(forest.score_samples(np.atleast_2d(vectors)), dtype=float)


class DepthForests:
    """One isolation forest per depth band, plus what the scorer needs to reach them."""

    def __init__(self, forests, band, space, scorer_covariance, dimension, terms=None,
                 params=None, n_intervals=0):
        self.forests = {int(depth): forest for depth, forest in forests.items()}
        self.band = int(band)
        self.space = space
        self.scorer_covariance = np.asarray(scorer_covariance, dtype=float)
        self._dimension = int(dimension)
        self._n_intervals = int(n_intervals)
        self.terms = list(terms) if terms is not None else None
        self.params = params or {}
        self.available_depths = np.array(sorted(self.forests))
        #: No unwhitened reference vectors are kept, so the off-manifold test cannot run
        #: against a forest. `StreamScorer` checks this and says so.
        self.raw = None

    @property
    def dimension(self):
        return self._dimension

    @property
    def reference_size(self):
        """Corpus intervals the forests were grown from - `PooledIndex`'s property, matched."""
        return self._n_intervals

    def depth_for_width(self, relative_width):
        """The closest indexed depth, on a log scale - `PooledIndex`'s rule, unchanged."""
        relative_width = min(max(float(relative_width), 1e-12), 1.0)
        target = -np.log2(relative_width)
        return int(self.available_depths[
            np.argmin(np.abs(self.available_depths - target))])

    def search(self, queries, depth, k=1, exclude_self=False):
        """Isolation score for queries already in this detector's space.

        Shaped like `PooledIndex.search` - ``(scores, neighbours)`` - so the scoring stage does
        not have to know which detector it holds. There is no neighbour to report, since nothing
        was matched against anything, so the second element is a column of -1.
        """
        queries = np.atleast_2d(np.asarray(queries, dtype=float))
        scores = forest_scores(self.forests[int(depth)], queries).reshape(-1, 1)
        return scores, np.full_like(scores, -1, dtype=int)

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            pickle.dump({"forests": self.forests, "band": self.band, "space": self.space,
                         "scorer_covariance": self.scorer_covariance,
                         "dimension": self._dimension, "terms": self.terms,
                         "params": self.params,
                         "n_intervals": self._n_intervals}, handle)
        return path

    @classmethod
    def load(cls, path):
        with Path(path).open("rb") as handle:
            stored = pickle.load(handle)
        return cls(stored["forests"], stored["band"], stored["space"],
                   stored["scorer_covariance"], stored["dimension"],
                   stored.get("terms"), stored.get("params"),
                   stored.get("n_intervals", 0))


def resolve_space(space):
    if space not in SPACES:
        raise ValueError(
            "detect.forest.space is {0!r}; expected one of {1}".format(space, list(SPACES)))
    return space


def prepare(corpus, covariance, space):
    """Read a corpus and put it in the requested space.

    Returns
    -------
    (np.ndarray, np.ndarray, np.ndarray, list, np.ndarray)
        The block to fit on, the depths, the source streams, the term names, and the matrix the
        scorer must apply to a fresh signature to reach the same space.
    """
    space = resolve_space(space)
    frame = read_corpus(corpus)
    if "depth" not in frame.columns:
        raise ValueError(
            "the corpus needs a 'depth' column - forests are fitted per depth band; got "
            "{0}".format(list(frame.columns)[:8]))

    signatures, terms = signature_matrix(frame)
    depths = frame["depth"].to_numpy(dtype=int)
    streams = frame[STREAM_COLUMN].to_numpy() if STREAM_COLUMN in frame.columns else None

    if space == "raw":
        # Identity, so the scorer's transform is a no-op and every detector reaches its own
        # space the same way. Wasteful by one matmul per query and worth it for having a
        # single code path.
        return signatures, depths, streams, terms, np.eye(signatures.shape[1])

    if isinstance(covariance, (str, Path)):
        covariance = load_covariance(covariance, expected_columns=terms)
    covariance = np.asarray(covariance, dtype=float)
    return signatures @ covariance.T, depths, streams, terms, covariance


def fit_forests(corpus, covariance, band=1, space="whitened",
                n_estimators=DEFAULT_N_ESTIMATORS, max_samples="auto", random_state=0,
                output_path=None, show_progress=False):
    """Fit one forest per depth, each on the intervals within `band` of that depth."""
    block, depths, _, terms, scorer_covariance = prepare(corpus, covariance, space)

    forests, sizes = {}, {}
    for depth in np.unique(depths):
        rows = np.flatnonzero(np.abs(depths - depth) <= band)
        forests[int(depth)] = make_forest(n_estimators, max_samples, random_state).fit(
            block[rows])
        sizes[int(depth)] = int(rows.size)

    detector = DepthForests(
        forests, band, space, scorer_covariance, block.shape[1], terms,
        params={"n_estimators": int(n_estimators), "max_samples": max_samples,
                "random_state": random_state, "per_depth_rows": sizes},
        n_intervals=block.shape[0])

    if show_progress:
        print("fitted {0} forest(s) of {1} tree(s) on {2}-dimensional {3} vectors, band "
              "{4}".format(len(forests), n_estimators, block.shape[1], space, band))
        for depth, count in sorted(sizes.items()):
            print("  depth {0}: {1:,} interval(s)".format(depth, count))

    if output_path is not None:
        detector.save(output_path)
        if show_progress:
            print("wrote {0}".format(output_path))
    return detector


def crossvalidated_forest_thresholds(corpus, covariance, k=5, statistic="p95", band=1,
                                     space="whitened", n_estimators=DEFAULT_N_ESTIMATORS,
                                     max_samples="auto", random_state=0, windowed=False,
                                     output_path=None, show_progress=False):
    """One threshold per depth, cross-validated over streams.

    The same protocol `crossvalidated_thresholding` applies to distances, and for the same
    reason: an interval scored against a forest that was fitted on its own stream is being asked
    how well that stream matches itself. Each fold is scored by a forest grown without it.
    """
    block, depths, streams, _, _ = prepare(corpus, covariance, space)
    if streams is None:
        raise ValueError(
            "the corpus has no {0!r} column, so folds cannot be formed over "
            "streams".format(STREAM_COLUMN))
    if windowed:
        streams = np.asarray([source_of(name) for name in streams])

    folds = stream_folds(streams, k, random_state)
    if show_progress:
        print("{0:,} interval(s) over {1} stream(s), {2} fold(s), band {3}, {4} space".format(
            len(block), len(np.unique(streams)), len(folds), band, space))

    collected = {int(depth): [] for depth in np.unique(depths)}
    for number, fold in enumerate(folds, start=1):
        held = np.isin(streams, fold)
        for depth in np.unique(depths[held]):
            queries = np.flatnonzero(held & (depths == depth))
            reference = np.flatnonzero(~held & (np.abs(depths - depth) <= band))
            if reference.size == 0:
                raise ValueError(
                    "fold {0} leaves no reference intervals within band {1} of depth {2}. "
                    "Lower k, or widen the band.".format(number, band, depth))
            forest = make_forest(n_estimators, max_samples, random_state).fit(block[reference])
            collected[int(depth)].extend(forest_scores(forest, block[queries]).tolist())
        if show_progress:
            print("  fold {0}/{1}: {2} stream(s), {3:,} interval(s) scored".format(
                number, len(folds), len(fold), int(held.sum())))

    reduce_to_threshold = parse_statistic(statistic)
    records = []
    for depth in sorted(collected):
        values = np.asarray(collected[depth], dtype=float)
        if not values.size:
            continue
        threshold = float(reduce_to_threshold(values))
        records.append({
            "depth": int(depth),
            "relative_width": 2.0 ** -int(depth),
            "n_calibration": int(values.size),
            "threshold": threshold,
            "median_distance": float(np.median(values)),
            "p90_distance": float(np.percentile(values, 90)),
            "p99_distance": float(np.percentile(values, 99)),
            "max_distance": float(values.max()),
            "n_above_threshold": int((values > threshold).sum()),
        })

    table = pd.DataFrame.from_records(records)
    thresholds = dict(zip(table["depth"].astype(int), table["threshold"].astype(float)))

    if show_progress:
        print("\n" + table.to_string(index=False))

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(output_path, index=False)
        if show_progress:
            print("wrote {0}".format(output_path))

    return thresholds, table
