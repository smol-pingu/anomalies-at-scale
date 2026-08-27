"""Pooled Mahalanobis kNN anomaly detection over dyadic intervals.

Takes a signature corpus of normality (as produced by
:mod:`anomalies_scale.signature_computer`), a set of streams to test, and an anomaly score
threshold, and returns the intervals of each stream that look anomalous.

The search is the widest-first dyadic segmentation from the SigNova paper's Appendix A: test
the whole stream first, and only bisect where a test fails, so clean stretches are cheap and
effort concentrates where there is something to find. Each candidate interval is scored by
its squared Mahalanobis distance to its nearest neighbour in the corpus, and an interval is
clean when that distance is at or below ``threshold``.

This generalises the toy notebook's ``pooled_anomaly_detector`` to arbitrary streams. Two
things had to change to get there:

**Intervals are matched by relative width, not absolute.** The corpus records each interval's
dyadic ``depth``, which corresponds to a relative width of ``2 ** -depth`` of its stream. A
test interval is assigned the depth whose relative width is closest on a log scale, so test
streams need not share a length with each other or with the corpus.

**Truncation is inferred, not assumed.** The corpus's signature-column count and the test
stream's channel count together determine the truncation level, so a corpus built at one
truncation cannot silently be used with streams signed at another.

A caveat on thresholds
----------------------
This module whitens with a single global ``Sigma^-1/2`` fitted across the whole corpus, and
compares one scalar ``threshold`` against raw squared Mahalanobis distances. Those two
choices interact badly if the corpus spans a wide range of interval widths.

A level-*k* signature term scales like (path increment) ** k, so wide intervals produce far
larger signature vectors than narrow ones. One global whitening does not remove that spread -
it applies the same linear map to every depth - so a distance of, say, 50 can be entirely
ordinary at depth 0 and wildly anomalous at depth 6. A single threshold then behaves like a
width filter rather than an anomaly test.

Fitting the whitening per depth would remove the problem, since each depth's vectors would be
normalised by their own covariance. That is not what this module does, so instead it measures
the spread: :func:`corpus_score_distribution` reports the corpus's own nearest-neighbour
distances per depth, and :class:`PooledCorpus` warns at construction if those scales diverge
enough that no single threshold can serve every depth. Run the diagnostic before trusting a
threshold you have not calibrated on this corpus.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import faiss
import iisignature
import numpy as np
import pandas as pd

from anomalies_scale.canonical_streams import expand_paths, load_stream
from anomalies_scale.signature_computer import add_base_point

#: Ratio between the largest and smallest per-depth median distance above which a single
#: scalar threshold is considered unsafe, and :class:`PooledCorpus` warns.
SCALE_SPREAD_WARNING = 10.0


# ---------------------------------------------------------------------------------------
# Corpus handling
# ---------------------------------------------------------------------------------------
def load_corpus(source):
    """Load a signature corpus as written by :func:`anomalies_scale.signature_computer.compute_corpus`.

    Accepts a parquet path or an already-loaded DataFrame, so a corpus can be built and used
    in one session without a round trip through disk.
    """
    if isinstance(source, pd.DataFrame):
        return source
    return pd.read_parquet(Path(source))


def split_corpus(frame):
    """Separate a corpus frame into its signature matrix and its depth labels.

    Returns
    -------
    (np.ndarray, np.ndarray)
        The ``(n_intervals, n_terms)`` signature matrix and the matching depth per row.
    """
    signature_cols = [c for c in frame.columns if c.startswith('sig_')]
    if not signature_cols:
        raise ValueError(
            "no 'sig_' columns found; expected a corpus from signature_computer, "
            'got columns {0}'.format(list(frame.columns)[:8]))
    if 'depth' not in frame.columns:
        raise ValueError("corpus is missing the 'depth' column")

    return (frame[signature_cols].to_numpy(dtype=float),
            frame['depth'].to_numpy(dtype=int))


def infer_truncation(width, n_terms):
    """Recover the truncation level from a channel count and a signature length.

    Guards against pairing a corpus with streams signed at a different truncation, which
    would otherwise surface as a confusing dimension mismatch deep inside FAISS.
    """
    for trunc in range(1, 32):
        length = iisignature.siglength(width, trunc)
        if length == n_terms:
            return trunc
        if length > n_terms:
            break
    raise ValueError(
        'no truncation level gives {0} signature terms for width {1}; the corpus and the '
        'test streams disagree about channel count or truncation'.format(n_terms, width))


def build_whitening(signatures, rcond=1e-12, chunk_rows=200_000):
    """Fit the global Mahalanobis whitening matrix ``Sigma^-1/2`` for a signature corpus.

    Returns the symmetric square root of the pseudo-inverse covariance, so plain Euclidean
    distance between whitened vectors is Mahalanobis distance::

        ||W(x - y)||^2 == (x - y)^T Sigma^-1 (x - y)

    Note this is *not* ``pinv(Sigma)``. Applying the inverse covariance directly to the
    differences - as some older code in this repository does - computes
    ``(x - y)^T Sigma^-2 (x - y)``, which weights every direction quadratically and, on data
    as ill-conditioned as signature vectors, can differ from the true Mahalanobis distance by
    many orders of magnitude.

    The fit never materialises the corpus as one float64 matrix: it makes a streaming pass for
    the mean, then folds centred chunks into a running QR factor. Stacking ``[R_prev; chunk]``
    and re-factorising is exact - R carries the same singular values and right singular vectors
    as the full stacked matrix - so this matches a direct fit to machine precision at
    ``O(chunk_rows * D)`` memory. QR rather than ``X^T X`` because forming the Gram matrix
    would square an already-brutal condition number.
    """
    signatures = np.asarray(signatures)
    n_rows, n_terms = signatures.shape
    if n_rows < 2:
        raise ValueError('need at least 2 corpus intervals to fit a covariance, got {0}'
                         .format(n_rows))

    total = np.zeros(n_terms, dtype=np.float64)
    for start in range(0, n_rows, chunk_rows):
        total += signatures[start:start + chunk_rows].sum(axis=0, dtype=np.float64)
    mu = total / n_rows

    R = np.zeros((0, n_terms), dtype=np.float64)
    for start in range(0, n_rows, chunk_rows):
        chunk = np.asarray(signatures[start:start + chunk_rows], dtype=np.float64) - mu
        R = np.linalg.qr(np.vstack([R, chunk]), mode='r')

    _, s, Vt = np.linalg.svd(R, full_matrices=False)

    scale = np.zeros_like(s)
    if s.size and s.max() > 0:
        keep = s > rcond * s.max()
        scale[keep] = np.sqrt(n_rows) / s[keep]

    return (Vt.T * scale) @ Vt


# ---------------------------------------------------------------------------------------
# The corpus, whitened and indexed
# ---------------------------------------------------------------------------------------
class PooledCorpus:
    """A signature corpus of normality, whitened once and indexed by depth band.

    Scoring an interval is one FAISS query against the corpus intervals whose depth lies
    within ``band`` of the query's own. Restricting the neighbour set to nearby scales keeps
    the comparison meaningful - a full-stream interval is not a useful neighbour for a short
    one - while including the adjacent depths keeps each reference set large enough for the
    nearest-neighbour distance to be stable.

    Because the whitening is global, the corpus is transformed exactly once and every band's
    index is a slice of that single whitened matrix. This is markedly cheaper than a
    per-depth whitening, which needs its own transformed copy of the reference set per depth.

    Parameters
    ----------
    corpus : pd.DataFrame or str or Path
        Signature corpus from :mod:`anomalies_scale.signature_computer`.
    covariance : np.ndarray, optional
        A precomputed ``Sigma^-1/2``. Fitted with :func:`build_whitening` if omitted.
        Note this is the whitening matrix, not the covariance itself.
    band : int
        How many depths either side of a query's own to include in its reference set.
        ``band=1`` searches depths ``d-1, d, d+1``; ``band=0`` restricts to same depth only.
    nn_reference_size : int, optional
        Cap on the number of corpus intervals held in the searchable index, sampled
        stratified across depths. Unlimited if omitted.
    random_state : int, optional
        Seed for that subsampling.
    check_scale : bool
        Measure the per-depth distance spread at construction and warn if a single scalar
        threshold cannot serve every depth. See the module docstring.
    """

    def __init__(self, corpus, covariance=None, band=1, nn_reference_size=None,
                 random_state=None, check_scale=True):
        frame = load_corpus(corpus)
        signatures, depths = split_corpus(frame)

        if nn_reference_size is not None and nn_reference_size < len(signatures):
            rng = np.random.default_rng(random_state)
            keep_fraction = nn_reference_size / len(signatures)
            selected = []
            for depth in np.unique(depths):
                rows = np.flatnonzero(depths == depth)
                take = max(1, int(round(keep_fraction * len(rows))))
                selected.append(rng.choice(rows, size=take, replace=False))
            selection = np.sort(np.concatenate(selected))
            signatures, depths = signatures[selection], depths[selection]

        self.n_terms = signatures.shape[1]
        self.depths = depths
        self.available_depths = np.unique(depths)
        self.band = int(band)

        self.covariance = (build_whitening(signatures) if covariance is None
                           else np.asarray(covariance, dtype=float))
        if self.covariance.shape != (self.n_terms, self.n_terms):
            raise ValueError('covariance must be ({0}, {0}) to match the corpus, got {1}'
                             .format(self.n_terms, self.covariance.shape))

        self._whitened = (signatures @ self.covariance.T).astype('float32')
        self._indices = {}

        if check_scale:
            self._warn_on_scale_spread()

    # -- internals ----------------------------------------------------------------------
    def _band_rows(self, depth):
        return np.flatnonzero(np.abs(self.depths - depth) <= self.band)

    def _index(self, depth):
        """FAISS index over the depth band around `depth`, built on first use."""
        if depth not in self._indices:
            rows = self._band_rows(depth)
            if rows.size == 0:
                raise ValueError('no corpus intervals within band {0} of depth {1}; corpus '
                                 'covers depths {2}'.format(self.band, depth,
                                                            list(self.available_depths)))
            index = faiss.IndexFlatL2(self.n_terms)
            index.add(np.ascontiguousarray(self._whitened[rows]))
            self._indices[depth] = index
        return self._indices[depth]

    def _warn_on_scale_spread(self):
        summary = self.score_distribution()
        medians = summary['median'].to_numpy()
        medians = medians[medians > 0]
        if medians.size < 2:
            return
        spread = medians.max() / medians.min()
        if spread > SCALE_SPREAD_WARNING:
            warnings.warn(
                'corpus nearest-neighbour distances span a factor of {0:.0f} across depths '
                '(median {1:.3g} to {2:.3g}). A single scalar threshold will behave partly '
                'as a width filter rather than an anomaly test. Inspect '
                'corpus_score_distribution() and consider a per-depth threshold.'.format(
                    spread, medians.min(), medians.max()),
                RuntimeWarning, stacklevel=3)

    # -- public -------------------------------------------------------------------------
    def depth_for_width(self, relative_width):
        """Assign an interval of the given relative width to the closest corpus depth.

        Compared on a log scale, since depth *d* corresponds to relative width ``2**-d`` and
        widths therefore halve from one depth to the next. Answering by scale rather than by
        tree position is what lets the search test intervals - such as those produced by
        boundary extension - that are not dyadic nodes at all.
        """
        relative_width = min(max(float(relative_width), 1e-12), 1.0)
        target = -np.log2(relative_width)
        return int(self.available_depths[
            np.argmin(np.abs(self.available_depths - target))])

    def score(self, signature, depth):
        """Squared Mahalanobis distance from `signature` to its nearest corpus neighbour."""
        query = np.asarray(signature, dtype=float).reshape(1, -1) @ self.covariance.T
        distances, _ = self._index(depth).search(np.ascontiguousarray(query, dtype='float32'), 1)
        return float(distances[0, 0])

    def score_distribution(self, sample_size=2000, random_state=0):
        """Per-depth distribution of the corpus's own nearest-neighbour distances.

        For each depth, samples corpus intervals and measures each one's distance to its
        nearest *other* corpus interval within the band. This is the empirical scale of
        "normal" at that depth, and so the basis for choosing a threshold - or for seeing
        that no single threshold will do.

        Returns
        -------
        pd.DataFrame
            One row per depth: ``n_intervals``, ``median``, ``p90``, ``p99``, ``max``.
        """
        rng = np.random.default_rng(random_state)
        rows = []

        for depth in self.available_depths:
            at_depth = np.flatnonzero(self.depths == depth)
            if at_depth.size == 0:
                continue
            sample = (rng.choice(at_depth, size=sample_size, replace=False)
                      if at_depth.size > sample_size else at_depth)

            queries = np.ascontiguousarray(self._whitened[sample])
            # k=2 and take the second: the nearest neighbour of a corpus point is itself.
            distances, _ = self._index(int(depth)).search(queries, 2)
            nearest = distances[:, 1]

            rows.append({'depth': int(depth), 'n_intervals': int(at_depth.size),
                         'median': float(np.median(nearest)),
                         'p90': float(np.percentile(nearest, 90)),
                         'p99': float(np.percentile(nearest, 99)),
                         'max': float(nearest.max())})

        return pd.DataFrame(rows)


def calibrate_threshold(corpus, statistic='p99', band=1, covariance=None,
                        output_path=None, **kwargs):
    """Derive a scalar anomaly threshold from the corpus's own score distribution.

    Takes the chosen `statistic` (``median``, ``p90``, ``p99`` or ``max``) at each depth and
    returns the largest across depths, so the threshold is permissive enough not to flag
    ordinary intervals at whichever depth is noisiest. Taking the maximum is deliberately
    conservative: with one global whitening the per-depth scales differ, and a threshold set
    from the tightest depth would flag everything at the loosest.

    That conservatism is also the weakness - if the per-depth scales differ a lot, no single
    number serves them all, and ``spread`` in the returned dict records by how much. Treat a
    spread above :data:`SCALE_SPREAD_WARNING` as a sign that per-depth thresholds are needed.

    Parameters
    ----------
    corpus : pd.DataFrame or str or Path
        Signature corpus of normality.
    statistic : str
        Column of :meth:`PooledCorpus.score_distribution` to threshold on.
    output_path : str or Path, optional
        JSON file to write the threshold and the full per-depth table to.

    Returns
    -------
    dict
        ``threshold``, ``statistic``, ``spread``, and ``per_depth`` (the table as records).
    """
    pooled = PooledCorpus(corpus, covariance=covariance, band=band, check_scale=False,
                          **kwargs)
    table = pooled.score_distribution()
    if statistic not in table.columns:
        raise ValueError('statistic must be one of {0}, got {1!r}'.format(
            [c for c in table.columns if c not in ('depth', 'n_intervals')], statistic))

    positive = table['median'][table['median'] > 0]
    spread = float(positive.max() / positive.min()) if len(positive) > 1 else 1.0

    result = {
        'threshold': float(table[statistic].max()),
        'statistic': statistic,
        'band': int(band),
        'spread': spread,
        'per_depth': table.to_dict(orient='records'),
    }

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as handle:
            json.dump(result, handle, indent=2)

    return result


def corpus_score_distribution(corpus, covariance=None, band=1, **kwargs):
    """Convenience wrapper: per-depth nearest-neighbour distance stats for a corpus.

    Use this to choose ``threshold``. A value between a depth's ``p99`` and its ``max`` flags
    only intervals more unusual than almost anything in the corpus; a value below the median
    flags most of them. If those figures differ wildly between depths, one threshold cannot
    serve them all - see the module docstring.
    """
    pooled = PooledCorpus(corpus, covariance=covariance, band=band, check_scale=False,
                          **kwargs)
    return pooled.score_distribution()


# ---------------------------------------------------------------------------------------
# Signing a test stream's intervals on demand
# ---------------------------------------------------------------------------------------
class IntervalSigner:
    """Base-pointed signatures of arbitrary sub-intervals of one stream.

    Unit intervals are signed once up front; any wider interval is composed from them by
    Chen's identity (``sigcombine``), memoising every composed node so overlapping queries
    reuse work. Because every unit interval is present, any ``lo < hi`` decomposes, so there
    is no fallback path.

    Set ``exact=True`` to compute each interval directly from its raw points instead. That is
    slower for long streams and many queries, but avoids the floating-point loss that repeated
    composition incurs on large unnormalised values - see
    the note in :mod:`anomalies_scale.signature_computer` on composition precision.
    """

    def __init__(self, path, trunc, exact=False):
        self.path = np.asarray(path, dtype=float)
        if self.path.ndim != 2 or self.path.shape[0] < 2:
            raise ValueError('path must be (length >= 2, width), got shape {0}'
                             .format(self.path.shape))
        self.trunc = trunc
        self.width = self.path.shape[1]
        self.exact = exact
        self._cache = {}
        if not exact:
            for i in range(self.path.shape[0] - 1):
                self._cache[(i, i + 1)] = iisignature.sig(self.path[i:i + 2], trunc)

    def _raw(self, lo, hi):
        if self.exact:
            return iisignature.sig(self.path[lo:hi + 1], self.trunc)
        if (lo, hi) in self._cache:
            return self._cache[(lo, hi)]
        mid = (lo + hi) // 2
        combined = iisignature.sigcombine(self._raw(lo, mid), self._raw(mid, hi),
                                          self.width, self.trunc)
        self._cache[(lo, hi)] = combined
        return combined

    def signature(self, lo, hi):
        """Base-pointed signature of the interval spanning points ``lo..hi`` inclusive."""
        return add_base_point(self._raw(lo, hi), self.path[lo], self.width, self.trunc)


# ---------------------------------------------------------------------------------------
# Widest-first dyadic search (SigNova paper, Appendix A)
# ---------------------------------------------------------------------------------------
def find_longest_clean(lo, hi, predicate, min_width):
    """Widest sub-interval of ``[lo, hi]`` satisfying `predicate`, by midpoint bisection.

    The whole range is tested first; only if that fails is it split, the left half searched
    before the right. Refuses to go narrower than ``min_width``, which is the search's
    resolution floor - anomalies shorter than that cannot be localised.
    """
    if hi - lo >= min_width and predicate(lo, hi):
        return (lo, hi)
    if hi - lo <= min_width:
        return None
    mid = (lo + hi) // 2
    return (find_longest_clean(lo, mid, predicate, min_width)
            or find_longest_clean(mid, hi, predicate, min_width))


def initial_extension_step(span, floor):
    """Largest power of two no greater than half the block's length, but at least `floor`.

    Extension reach scales with what has already been proven clean: a long clean run is
    entitled to reach a long way on its first attempt, a short one is not. The floor keeps a
    block at the resolution limit from being frozen - half of a 4-point block is 2, below the
    floor, and without this such a block could never extend at all.
    """
    half = max(span // 2, 1)
    return max(1 << (half.bit_length() - 1), floor)


def extend_boundary(block_lo, block_hi, limit, predicate, floor, extend_low):
    """Binary-search the furthest boundary that still passes, on one side of a clean block.

    Starts from a step of :func:`initial_extension_step` and **halves it on every failure,
    never raising it again**, stopping once it would fall below `floor`. A success keeps the
    current step and tries the same distance again from the new edge.

    That schedule reaches a distant boundary in ``O(log n)`` predicate calls where a fixed
    step needs ``O(n)``, and it converges on the true boundary rather than stopping at the
    last multiple of some fixed increment. Intervals do get retested - a step of 4 taken twice
    lands on the same place a failed step of 8 tried - but both detectors memoise the
    predicate by ``(lo, hi)``, so a repeat costs a dictionary lookup rather than a model.
    """
    step = initial_extension_step(block_hi - block_lo, floor)
    while step >= floor:
        if extend_low:
            candidate = block_lo - step
            passes = candidate >= limit and predicate(candidate, block_hi)
        else:
            candidate = block_hi + step
            passes = candidate <= limit and predicate(block_lo, candidate)

        if passes:
            if extend_low:
                block_lo = candidate
            else:
                block_hi = candidate
        else:
            # Failure halves the step - which also covers a candidate that simply ran past
            # the region bound, since a shorter reach may still fit.
            step //= 2

    return block_lo if extend_low else block_hi


def widest_first_segment(lo_bound, hi_bound, predicate, min_width, floor=None):
    """Segment ``[lo_bound, hi_bound]`` into clean intervals, widest first.

    Repeatedly finds the widest clean interval anywhere in the current region, extends its
    boundaries outward by binary search for as long as they still pass, records it, and
    recurses into the untouched regions on either side.

    ``floor`` is the finest extension step, and defaults to ``min_width`` - the search's own
    resolution limit, on the grounds that a boundary cannot be meaningfully placed finer than
    the narrowest interval the predicate is willing to judge.
    """
    floor = min_width if floor is None else floor
    found, stack = [], [(lo_bound, hi_bound)]

    while stack:
        lo, hi = stack.pop()
        if hi <= lo:
            continue

        block = find_longest_clean(lo, hi, predicate, min_width)
        if block is None:
            continue  # nothing clean anywhere in this piece

        block_lo, block_hi = block
        block_lo = extend_boundary(block_lo, block_hi, lo, predicate, floor, extend_low=True)
        # Right extension starts from the already-extended left edge, so the two sides are
        # not judged on identical terms - the right is tested against a block the left has
        # already widened.
        block_hi = extend_boundary(block_lo, block_hi, hi, predicate, floor, extend_low=False)

        found.append((block_lo, block_hi))
        if block_lo > lo:
            stack.append((lo, block_lo))
        if block_hi < hi:
            stack.append((block_hi, hi))

    return found


def invert_clean(clean, last_index):
    """Complement of the clean intervals - the anomalous ranges, in SigNova's convention."""
    anomalous, cursor = [], 0
    for lo, hi in sorted(clean):
        if lo > cursor:
            anomalous.append((cursor, lo))
        cursor = max(cursor, hi)
    if cursor < last_index:
        anomalous.append((cursor, last_index))
    return anomalous


#: Kept so existing callers of the private name keep working; the search internals are worth
#: being public, since the whole point of the widest-first procedure is that what counts as
#: "clean" is somebody else's decision.
_invert = invert_clean


# ---------------------------------------------------------------------------------------
# The detector
# ---------------------------------------------------------------------------------------
def detect_stream_anomalies(path, pooled, threshold, trunc=None, tol=0, sig_tol=2,
                            exact_signatures=False):
    """Find the anomalous intervals of a single stream.

    Parameters
    ----------
    path : np.ndarray
        ``(length, width)`` with time in the first column.
    pooled : PooledCorpus
        The corpus of normality to score against.
    threshold : float
        Squared Mahalanobis distance above which an interval is anomalous.
    trunc : int, optional
        Truncation level. Inferred from the corpus's term count and the stream's width if
        omitted, which also catches a corpus and stream that disagree about either.

    Returns
    -------
    (list, dict)
        ``[(lo, hi, score), ...]`` in point indices, and a stats dict recording how many
        candidate intervals the search evaluated.
    """
    if trunc is None:
        trunc = infer_truncation(path.shape[1], pooled.n_terms)
    signer = IntervalSigner(path, trunc, exact=exact_signatures)
    last = path.shape[0] - 1
    span = float(last)
    stats = {'evaluated': 0}

    def is_clean(lo, hi):
        lo, hi = max(0, lo), min(last, hi)
        stats['evaluated'] += 1
        depth = pooled.depth_for_width((hi - lo) / span)
        return pooled.score(signer.signature(lo, hi), depth) <= threshold

    # The extension floor is the search's own resolution limit, not `tol`. Extension is a
    # binary search now rather than a fixed-step walk, so there is no increment to configure -
    # `tol` survives only as an explicit override for callers that want a coarser floor.
    clean = widest_first_segment(0, last, is_clean, 2 ** sig_tol,
                                 floor=2 ** tol if tol else None)

    scored = []
    for lo, hi in _invert(clean, last):
        depth = pooled.depth_for_width((hi - lo) / span)
        scored.append((lo, hi, pooled.score(signer.signature(lo, hi), depth)))
    return scored, stats


def detect_anomalies(streams, corpus, threshold, covariance=None, band=1,
                     tol=0, sig_tol=2, nn_reference_size=None, random_state=None,
                     output_path=None, time_col=None, value_cols=None,
                     exact_signatures=False, show_progress=False):
    """Run pooled Mahalanobis kNN anomaly detection over a collection of streams.

    Parameters
    ----------
    streams : iterable
        Stream files (parquet/CSV, read by :func:`canonical_streams.load_stream`) or
        ``(length, width)`` arrays with time in the first column. The two may be mixed.
    corpus : pd.DataFrame or str or Path
        Signature corpus of normality from :mod:`anomalies_scale.signature_computer`.
    threshold : float
        Squared Mahalanobis distance above which an interval is anomalous. Choose it with
        :func:`corpus_score_distribution`; read the module docstring first, because a single
        scalar is only safe when the corpus's per-depth distance scales are comparable.
    covariance : np.ndarray, optional
        Precomputed whitening matrix ``Sigma^-1/2``. Fitted from the corpus if omitted.
    band : int
        Depth band searched around each query's own depth. See :class:`PooledCorpus`.
    tol, sig_tol : int
        Boundary extension step and resolution floor, as powers of two in points. ``sig_tol=2``
        means the search never tests an interval narrower than 4 points, and so cannot
        localise anything shorter.
    nn_reference_size, random_state :
        Cap and seed for subsampling the searchable corpus.
    output_path : str or Path, optional
        Parquet to write the result to.
    time_col, value_cols :
        Passed to :func:`canonical_streams.load_stream` for file inputs.
    exact_signatures : bool
        Compute each interval's signature directly rather than composing it via Chen's
        identity. Slower, but avoids composition's floating-point loss on large values.
    show_progress : bool
        Print per-stream progress.

    Returns
    -------
    pd.DataFrame
        One row per flagged interval: ``stream``, ``start_index``, ``end_index``,
        ``start_time``, ``end_time``, ``depth``, ``score``. Empty (with those columns) if
        nothing is flagged.
    """
    streams = list(streams)
    if not streams:
        raise ValueError('no streams given')

    pooled = PooledCorpus(corpus, covariance=covariance, band=band,
                          nn_reference_size=nn_reference_size, random_state=random_state)

    records = []
    for number, stream in enumerate(streams, start=1):
        if isinstance(stream, (str, Path)):
            name = Path(stream).stem
            path = load_stream(stream, time_col=time_col, value_cols=value_cols)
        else:
            name = 'stream_{0}'.format(number - 1)
            path = np.asarray(stream, dtype=float)

        intervals, stats = detect_stream_anomalies(
            path, pooled, threshold, trunc=infer_truncation(path.shape[1], pooled.n_terms),
            tol=tol, sig_tol=sig_tol, exact_signatures=exact_signatures)

        span = float(path.shape[0] - 1)
        for lo, hi, score in intervals:
            records.append({
                'stream': name,
                'start_index': int(lo), 'end_index': int(hi),
                'start_time': float(path[lo, 0]), 'end_time': float(path[hi, 0]),
                'depth': pooled.depth_for_width((hi - lo) / span),
                'score': score,
            })

        if show_progress:
            print('[{0}/{1}] {2}: {3} flagged interval(s), {4} candidates evaluated'.format(
                number, len(streams), name, len(intervals), stats['evaluated']))

    frame = pd.DataFrame(records, columns=['stream', 'start_index', 'end_index',
                                           'start_time', 'end_time', 'depth', 'score'])

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(output_path, index=False)
        if show_progress:
            print('wrote {0} flagged interval(s) to {1}'.format(len(frame), output_path))

    return frame


def main(argv=None):
    """Command line entry point."""
    parser = argparse.ArgumentParser(
        description='Pooled Mahalanobis kNN anomaly detection over dyadic intervals.')
    parser.add_argument('streams', nargs='+',
                        help='stream files to test, or wildcard patterns (expanded here, so '
                             'they work on shells that do not expand them themselves)')
    parser.add_argument('--corpus', required=True, help='signature corpus parquet')
    parser.add_argument('--threshold', type=float,
                        help='squared Mahalanobis distance above which an interval is anomalous')
    parser.add_argument('--band', type=int, default=1,
                        help='depth band searched either side of a query (default 1)')
    parser.add_argument('--sig-tol', type=int, default=2,
                        help='resolution floor as a power of two, in points (default 2)')
    parser.add_argument('--tol', type=int, default=0,
                        help='boundary extension step as a power of two (default 0)')
    parser.add_argument('--nn-reference-size', type=int, default=None,
                        help='cap on searchable corpus intervals')
    parser.add_argument('--output', default=None, help='parquet to write flagged intervals to')
    parser.add_argument('--time-col', default=None, help='name of the time column')
    parser.add_argument('--exact-signatures', action='store_true',
                        help='compute signatures directly instead of composing them')
    parser.add_argument('--describe', action='store_true',
                        help='print the corpus per-depth score distribution and exit, to '
                             'help choose a threshold')
    parser.add_argument('--calibrate-out', default=None,
                        help='derive a threshold from the corpus, write it as JSON, and exit')
    parser.add_argument('--statistic', default='p99',
                        help='per-depth statistic to calibrate on (default p99)')
    parser.add_argument('--quiet', action='store_true', help='suppress progress output')

    args = parser.parse_args(argv)

    if args.describe:
        print(corpus_score_distribution(args.corpus, band=args.band).to_string(index=False))
        return

    if args.calibrate_out is not None:
        result = calibrate_threshold(args.corpus, statistic=args.statistic, band=args.band,
                                     output_path=args.calibrate_out)
        if not args.quiet:
            print('threshold ({0}) = {1:.6g}, per-depth median spread {2:.1f}x -> {3}'.format(
                result['statistic'], result['threshold'], result['spread'],
                args.calibrate_out))
        return

    if args.threshold is None:
        parser.error('--threshold is required unless --describe is given')

    detect_anomalies(
        expand_paths(args.streams), args.corpus, threshold=args.threshold, band=args.band,
        tol=args.tol, sig_tol=args.sig_tol, nn_reference_size=args.nn_reference_size,
        output_path=args.output, time_col=args.time_col,
        exact_signatures=args.exact_signatures, show_progress=not args.quiet)


if __name__ == '__main__':
    main()
