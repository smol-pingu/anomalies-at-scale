"""Score canonical test streams against a corpus index, and report their anomalous intervals.

Takes a canonical stream set, the FAISS index built by :mod:`anomalies_scale.index_creation`,
the ``Sigma^-1/2`` that index was whitened with, and a threshold. Returns the same frame with
each stream's anomalous intervals attached.

The search
----------
Widest-first, as in SigNova's Appendix A. The whole stream is tested first; if it fails, it is
bisected and each half tested, and so on down to a resolution floor. Whatever is found clean is
then grown outward by binary search - try to extend by half the block, then a quarter, and so
on - so a clean region is described by as few, as wide intervals as possible. What is reported
is the **complement**: the points no clean interval covers.

That is why this is not a per-point classifier. The question asked of the corpus is always
"does this *interval* look like normal behaviour", which is what a signature can answer;
per-point labels fall out of the intervals afterwards.

Scoring one interval
--------------------
Four steps, and each must match how the corpus was built or the distance means nothing:

1. slice the stream to ``[lo, hi]`` and sign it, base-pointed, exactly as
   :mod:`anomalies_scale.signature_computer` signed the corpus;
2. apply the fitted normaliser, if the corpus was normalised;
3. multiply by ``Sigma^-1/2``, which is what makes Euclidean distance in the index *be*
   Mahalanobis distance;
4. query the index at the interval's own depth band and take the nearest-neighbour distance.

Step 2 needs the normaliser **fitted on the corpus**, not one fitted here. Re-deriving centre
and scale from test data would put the two sides in different units, and the index and the
covariance both live in the corpus's units.

Thresholds
----------
Either a scalar or a per-depth table. Per-depth is what
:mod:`anomalies_scale.threshold_calibration` produces and the better-founded choice: a level-*k*
signature term scales like the increment to the *k*-th power, so wide intervals carry
systematically larger vectors than narrow ones, and one number for all of them behaves partly
as a width filter rather than as an anomaly test.

Interval convention
-------------------
``[lo, hi]`` are **inclusive** point indices into the stream as stored - position, not time.
This matches the rest of the project, where adjacent intervals share a boundary point so that
composing them introduces no discontinuity.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from anomalies_scale.anomaly_detection_pooled import (
    infer_truncation, invert_clean, widest_first_segment)
from anomalies_scale.canonical_streams import STREAM_COLUMN, read_canonical
from anomalies_scale.signature_computer import base_pointed_signature, stream_paths

#: Column added to the returned frame: the anomalous intervals, as ``[[lo, hi], ...]``.
ANOMALOUS_COLUMN = "anomalous"

#: Column names a per-depth threshold table might use for its value.
THRESHOLD_VALUE_COLUMNS = ("threshold", "value", "p99", "score")


def resolve_threshold(threshold):
    """Turn a scalar, a mapping, a DataFrame or a CSV path into a ``depth -> float`` lookup.

    A depth with no entry of its own falls back to the nearest depth that has one, rather than
    raising. The bands either side of a query are already searched together, so a threshold one
    depth away is the right order of magnitude; refusing to score would be a worse answer than
    a slightly stale one.
    """
    if isinstance(threshold, (int, float, np.integer, np.floating)):
        value = float(threshold)
        return (lambda depth: value), {"kind": "scalar", "value": value}

    if isinstance(threshold, dict):
        table = {int(k): float(v) for k, v in threshold.items()}
    else:
        frame = (threshold if isinstance(threshold, pd.DataFrame)
                 else pd.read_csv(Path(threshold)))
        if "depth" not in frame.columns:
            raise ValueError(
                "a threshold table needs a 'depth' column; got {0}".format(
                    list(frame.columns)[:8]))
        value_column = next((c for c in THRESHOLD_VALUE_COLUMNS if c in frame.columns), None)
        if value_column is None:
            raise ValueError(
                "no threshold column found in {0}; expected one of {1}".format(
                    list(frame.columns)[:8], list(THRESHOLD_VALUE_COLUMNS)))
        table = dict(zip(frame["depth"].astype(int), frame[value_column].astype(float)))

    if not table:
        raise ValueError("the threshold table is empty")

    def lookup(depth):
        depth = int(depth)
        if depth in table:
            return table[depth]
        return table[min(table, key=lambda known: abs(known - depth))]

    return lookup, {"kind": "per_depth", "depths": sorted(table)}


class StreamScorer:
    """Scores intervals of one stream at a time against a fitted corpus index.

    Holds everything that must not vary between intervals - the index, the metric, the
    normaliser, the truncation - so that the only thing changing per query is which points are
    being asked about.
    """

    def __init__(self, index, covariance, threshold, normaliser=None, trunc=None, span=None,
                 subspace=None, subspace_threshold=None, signature_projection=None,
                 neighbours=1):
        from anomalies_scale.covariance_creation import Whitening

        self.index = index
        self.covariance = (covariance if isinstance(covariance, Whitening)
                           else Whitening.from_dense(np.asarray(covariance, dtype=float)))
        self.threshold, self.threshold_info = resolve_threshold(threshold)
        self.normaliser = normaliser
        #: Which neighbour's distance is the score - 1 the nearest, n the n-th nearest. Must
        #: match what the threshold was calibrated at; `crossvalidated_thresholding` takes the
        #: same argument and says why.
        self.neighbours = int(neighbours)
        #: Maps a full-width signature into the latent space the corpus was reduced to. When
        #: this is set the index no longer holds signatures, so `truncation` cannot infer the
        #: level from the index width and `trunc` must be given.
        self.signature_projection = signature_projection
        self.trunc = trunc
        self.span = int(span) if span else None
        self.subspace = None if subspace is None else np.asarray(subspace, dtype=float)
        self.subspace_threshold = (None if subspace_threshold is None
                                   else float(subspace_threshold))
        self.n_queries = 0
        self.n_off_manifold = 0

        if self.off_manifold_enabled and self.index.raw is None:
            raise ValueError(
                "the off-manifold test needs the corpus vectors as they were before whitening, "
                "and this index was built without them. Rebuild with detect.off_manifold set, "
                "which makes `index` store them.")

    @property
    def off_manifold_enabled(self):
        """Whether the residual test is armed."""
        return self.subspace is not None and self.subspace_threshold is not None

    def off_manifold(self, raw, neighbour_row):
        """Whether a query lies outside the corpus's row space, and by how much.

        SigMahaKNN's ``subspace_thres`` test, from ``Mahalanobis.calc_distance``: take the
        difference between the query and the corpus point it matched, and measure what fraction
        of its norm survives projection *out* of the corpus row space::

            rho = ||x - x V^T V|| / ||x||

        Above the threshold the point is one the corpus cannot represent at all, and the
        Mahalanobis distance is not the right question - a direction with no corpus variance is
        annihilated by the pseudo-inverse, so an interval moving only in such directions scores
        near zero however unlike the corpus it is. That is the failure this closes.

        On the difference rather than on the query alone, because the corpus rows lie in the
        row space by construction: the neighbour contributes nothing to the numerator, but its
        distance sets the scale of the denominator, which is what makes 1e-3 a meaningful
        number rather than an arbitrary one.
        """
        difference = raw.ravel() - self.index.raw[int(neighbour_row)]
        norm = float(np.linalg.norm(difference))
        if norm <= 0.0:
            return False, 0.0

        residual = difference - difference @ self.subspace.T @ self.subspace
        rho = float(np.linalg.norm(residual) / norm)
        return rho > self.subspace_threshold, rho

    def truncation(self, width):
        """The truncation level, inferred from the index width if not given.

        Inferring rather than trusting a parameter is deliberate: it is the one check that the
        streams being scored have the same channel count the corpus was built at. A mismatch
        would otherwise surface as a FAISS dimension error with no explanation.
        """
        if self.trunc is None:
            if self.signature_projection is not None:
                raise ValueError(
                    "the corpus was reduced to a latent signature space, so the index width no "
                    "longer says what truncation it was built at. Pass `trunc` explicitly.")
            # The number of *signature terms*, not the index width. Under a factored metric the
            # index holds r latent coordinates and inferring from that would give the wrong
            # truncation - silently, since a plausible level would come back. `terms` carries
            # the signature column names and so is D whatever the index stores.
            terms = getattr(self.index, "terms", None)
            self.trunc = infer_truncation(
                width, len(terms) if terms else self.index.dimension)
        return self.trunc

    def signature(self, path, lo, hi):
        """One interval's signature, in the corpus's units and in the index's coordinates.

        Returns both forms. The whitened vector is what the index is searched with; the
        unwhitened one is what the off-manifold test needs, since ``Sigma^-1/2`` projects onto
        the retained subspace and destroys the very component that test looks for.
        """
        raw = base_pointed_signature(path[lo:hi + 1], self.truncation(path.shape[1]))
        vector = np.asarray(raw, dtype=float).reshape(1, -1)

        if self.normaliser is not None:
            vector = (vector - self.normaliser.centre) / self.normaliser.scale
        if self.signature_projection is not None:
            # Into the latent space the corpus was reduced to, before the metric - which was
            # fitted on latent vectors and has no meaning applied to a full-width one.
            vector = np.asarray(self.signature_projection.transform(vector), dtype=float)
        return vector, self.covariance.apply(vector)

    def score(self, path, lo, hi):
        """Squared Mahalanobis distance from interval ``[lo, hi]`` to its `neighbours`-th normal.

        Depth comes from the interval's width relative to `span`. That is the stream's own
        length by default, which is right when whole streams are scored against a corpus built
        from whole streams - both sides are then measured against their own extent.

        Under windowing it is not. The corpus holds intervals of a *fixed* number of points, and
        a short final window measured against itself would report a full-width interval and land
        at depth 0 - where it would be judged against references three or four times its length,
        score low for that reason alone, and read as clean. On run-to-failure data that window
        is the failure. Passing the standard window size as `span` puts it in the band its
        absolute length belongs to instead.
        """
        span = float(self.span if self.span else path.shape[0] - 1)
        depth = self.index.depth_for_width((hi - lo) / span if span else 1.0)
        self.n_queries += 1

        raw, whitened = self.signature(path, lo, hi)
        distances, neighbours = self.index.search(whitened, depth, k=self.neighbours)
        # The last column: FAISS returns neighbours in ascending distance, so column n - 1 is
        # the n-th nearest. Detectors that answer with a single score - a forest, an LOF -
        # ignore `k` and return one column, and the same indexing picks it up unchanged.
        value = float(np.asarray(distances)[0, -1])

        if self.off_manifold_enabled:
            # The same neighbour the distance was taken from, so the residual test is asked
            # about the match actually being reported rather than a nearer one that was not.
            outside, _ = self.off_manifold(raw, np.asarray(neighbours)[0, -1])
            if outside:
                # Infinite, not merely large. The point is not far from the corpus within the
                # space the corpus describes - it is outside that space, and any finite number
                # would be a claim about a geometry that does not apply to it.
                self.n_off_manifold += 1
                return float("inf"), depth

        return value, depth

    def scan(self, path, sig_tol=2, tol=0):
        """Widest-first search over one stream, returning its anomalous intervals.

        Returns
        -------
        (list, list)
            ``[[lo, hi], ...]`` inclusive point indices, and ``(lo, hi, score)`` per
            anomalous interval.
        """
        last = path.shape[0] - 1
        if last < 1:
            return [], []

        def is_clean(lo, hi):
            lo, hi = max(0, int(lo)), min(last, int(hi))
            if hi - lo < 1:
                return True
            value, depth = self.score(path, lo, hi)
            return value <= self.threshold(depth)

        clean = widest_first_segment(0, last, is_clean, 2 ** sig_tol,
                                     floor=2 ** tol if tol else None)

        intervals, scored = [], []
        for lo, hi in invert_clean(clean, last):
            intervals.append([int(lo), int(hi)])
            scored.append((int(lo), int(hi), self.score(path, int(lo), int(hi))[0]))
        return intervals, scored


def load_subspace(source, expected_columns=None):
    """Read the numerical-rank basis written beside the covariance.

    Returns ``(rank, n_terms)`` with orthonormal rows. The term names travel with it and are
    checked, because a basis applied to a corpus whose columns are in a different order would
    project onto the wrong directions and fail silently.
    """
    if source is None:
        return None
    if isinstance(source, np.ndarray):
        return np.asarray(source, dtype=float)

    stored = np.load(Path(source), allow_pickle=False)
    basis = np.asarray(stored["basis"], dtype=float)

    if expected_columns is not None and "columns" in stored.files:
        columns = [str(c) for c in stored["columns"]]
        if columns != list(expected_columns):
            raise ValueError(
                "the off-manifold basis was fitted on different signature terms than the index "
                "holds ({0} vs {1} term(s)); they cannot be used together".format(
                    len(columns), len(expected_columns)))
    return basis


def score_streams(streams, index, covariance, threshold, normaliser=None, trunc=None,
                  sig_tol=2, tol=0, span=None, subspace=None, subspace_threshold=None,
                  signature_projection=None, neighbours=1, output_path=None,
                  stats_path=None, show_progress=False):
    """Score every stream in a canonical set and attach its anomalous intervals.

    Parameters
    ----------
    streams : path or pd.DataFrame
        Canonical stream set - ``stream``, ``Time``, ``Stream``.
    index : PooledIndex
        From :func:`anomalies_scale.index_creation.load_index`.
    covariance : path or np.ndarray or pd.DataFrame
        The same ``Sigma^-1/2`` the index was whitened with. A different one would place test
        signatures in a space the index does not live in.
    threshold : float, dict, pd.DataFrame or path
        A scalar for every depth, or a per-depth table.
    normaliser : Normaliser or path, optional
        Fitted on the corpus. Omit when the corpus was not normalised.
    span : int, optional
        Increments a full-width interval spans, for the purpose of choosing a depth band. Pass
        `window.size` when the streams are windows; leave it out when they are whole streams.
        See :meth:`StreamScorer.score`.
    neighbours : int
        Which neighbour's distance is the score - 1 the nearest, n the n-th. Must match the k
        the threshold was calibrated at, or the two describe different distributions.
    stats_path : path, optional
        JSON to write the scoring pass's counts to - streams, points, and the interval queries
        the search actually issued. `anomalies_scale.throughput` divides the benchmark's seconds
        by these; without them a stage time says nothing about throughput. The per-stream query
        counts are kept as a distribution rather than a mean, because the widest-first search
        only bisects where a test fails, so an anomalous stream costs more than a clean one.

    Returns
    -------
    pd.DataFrame
        The canonical frame with ``anomalous`` added - ``[[lo, hi], ...]`` inclusive point
        indices - plus per-stream counts and the worst score found. Any extra columns the input
        carried, such as the window bookkeeping, are preserved.
    """
    frame = read_canonical(streams)

    if isinstance(covariance, (str, Path)):
        from anomalies_scale.covariance_creation import load_whitening

        # Keeps the factor when the metric was stored factored, so a query is projected into the
        # same r coordinates the index holds rather than multiplied out to full width.
        covariance = load_whitening(covariance)
    elif isinstance(covariance, pd.DataFrame):
        covariance = covariance.to_numpy()
    # `StreamScorer` wraps anything that is not already a Whitening, so a bare array from a
    # caller keeps working exactly as before.

    if isinstance(normaliser, (str, Path)):
        from anomalies_scale.normalisation import Normaliser

        normaliser = Normaliser.load(normaliser)

    if isinstance(subspace, (str, Path)):
        subspace = load_subspace(subspace, expected_columns=index.terms)

    if isinstance(signature_projection, (str, Path)):
        from anomalies_scale.umap_projection import Projection

        signature_projection = Projection.load(signature_projection)

    scorer = StreamScorer(index, covariance, threshold, normaliser, trunc, span,
                          subspace, subspace_threshold, signature_projection, neighbours)
    if show_progress:
        print("scoring {0} stream(s) against {1} reference vector(s); threshold {2}".format(
            len(frame), scorer.index.reference_size, scorer.threshold_info))
        if scorer.neighbours != 1:
            print("  scoring against the {0}-th nearest neighbour, not the first".format(
                scorer.neighbours))
        if normaliser is None:
            print("  signatures are NOT normalised - correct only if the corpus was not")
        if scorer.span:
            print("  depth bands taken against a {0}-increment window, not each stream's own "
                  "length".format(scorer.span))
        if scorer.off_manifold_enabled:
            print("  off-manifold test on, rho > {0:g} against a {1}-dimensional corpus row "
                  "space".format(scorer.subspace_threshold, scorer.subspace.shape[0]))
        if scorer.signature_projection is not None:
            print("  signatures projected into a {0}-dimensional latent space before "
                  "scoring".format(scorer.signature_projection.n_components))

    rows, per_stream_queries, n_points = [], [], 0
    started = time.time()
    for number, (name, path, _) in enumerate(stream_paths(frame), start=1):
        # Queries are counted per stream by difference, not just in total: the widest-first
        # search bisects only where a test fails, so its cost depends on how anomalous the data
        # is. That is a distribution worth keeping, not a mean worth reporting.
        before = scorer.n_queries
        intervals, scored = scorer.scan(path, sig_tol=sig_tol, tol=tol)
        per_stream_queries.append(int(scorer.n_queries - before))
        n_points += int(path.shape[0])
        flagged = int(sum(hi - lo + 1 for lo, hi in intervals))
        rows.append({
            ANOMALOUS_COLUMN: intervals,
            "n_anomalous_intervals": len(intervals),
            "n_anomalous_points": flagged,
            "anomalous_fraction": flagged / path.shape[0],
            "max_score": max((s for _, _, s in scored), default=0.0),
        })
        if show_progress and number % 25 == 0:
            print("  scored {0}/{1} stream(s)".format(number, len(frame)))

    out = frame.copy()
    for column in rows[0] if rows else []:
        out[column] = [row[column] for row in rows]

    if show_progress:
        print("{0} of {1} stream(s) carry a flag; {2:.2%} of points flagged overall; "
              "{3:,} interval queries".format(
                  int((out["n_anomalous_intervals"] > 0).sum()), len(out),
                  float(out["anomalous_fraction"].mean()), scorer.n_queries))
        if scorer.off_manifold_enabled:
            print("  {0:,} of {1:,} queries ({2:.2%}) were off-manifold".format(
                scorer.n_off_manifold, scorer.n_queries,
                scorer.n_off_manifold / max(scorer.n_queries, 1)))

    if stats_path is not None:
        import json

        from anomalies_scale.throughput import peak_memory_bytes

        queries = np.asarray(per_stream_queries, dtype=float)
        stats = {
            "n_streams": int(len(frame)),
            # Points the scorer traversed. Under windowing this exceeds the observation count
            # the metrics are measured over, because adjacent windows share a boundary point -
            # 21,866 against 20,631 on FD001, about 6%.
            "n_points": int(n_points),
            "peak_rss_bytes": peak_memory_bytes(),
            "n_queries": int(scorer.n_queries),
            "n_off_manifold": int(scorer.n_off_manifold),
            "reference_size": int(scorer.index.reference_size),
            "dimension": int(scorer.index.dimension),
            "neighbours": int(scorer.neighbours),
            "seconds": round(time.time() - started, 3),
            # The shape of the per-stream cost, since the mean hides it.
            "queries_per_stream": {
                "min": int(queries.min()) if queries.size else 0,
                "median": float(np.median(queries)) if queries.size else 0.0,
                "mean": float(queries.mean()) if queries.size else 0.0,
                "p90": float(np.percentile(queries, 90)) if queries.size else 0.0,
                "max": int(queries.max()) if queries.size else 0,
            },
        }
        stats_path = Path(stats_path)
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
        if show_progress:
            print("  wrote scoring counts to {0}".format(stats_path))

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        storable = out.copy()
        for column in ("Time", "Stream"):
            if column in storable.columns:
                storable[column] = [np.asarray(v).tolist() for v in storable[column]]
        storable[ANOMALOUS_COLUMN] = [
            [[int(lo), int(hi)] for lo, hi in v] for v in storable[ANOMALOUS_COLUMN]]
        storable.to_parquet(output_path, index=False)

    return out
