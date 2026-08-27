"""Whiten a signature corpus and build the searchable FAISS index over it.

Takes the corpus of normality and the ``Sigma^-1/2`` fitted by
:mod:`anomalies_scale.covariance_creation`, applies one to the other, and writes a flat L2
index over the result. Whitening first is what makes plain Euclidean distance *be*
Mahalanobis distance::

    ||W(x - y)|| ** 2  ==  (x - y) @ Sigma^-1 @ (x - y).T

which is the only reason a FAISS ``IndexFlatL2`` is the right structure here. The matrix
must therefore be the square root form; :func:`build_index` refuses anything else it can
detect, because passing ``Sigma^-1`` in instead silently computes ``Sigma^-2`` distances and
the error is enormous on data this ill-conditioned.

What the index contains
-----------------------
The **fit and withheld** intervals both. The metric was fitted on `fit` alone, but the
searchable set is deliberately larger: withheld windows are normal data, and coverage of
normality is exactly what a nearest-neighbour detector is short of. Which rows came from
which split is recorded alongside, because calibration has to score the withheld intervals
against this very index while skipping each one's own zero-distance self-match.

Depth banding
-------------
A query is only compared against corpus intervals whose dyadic depth is within ``band`` of
its own. Restricting to nearby scales keeps the comparison meaningful - a whole-stream
interval is not a useful neighbour for a short one - while including the adjacent depths
keeps each reference set large enough for the nearest-neighbour distance to be stable.

FAISS 1.15 offers no filtered search over a flat index, so the band has to be structural.
Rather than write one index per band - which duplicates every vector into two or three of
them - a single flat index is written and the per-band views are derived on load with
``reconstruct_n``, which is exact and costs a memcpy. The artifact stays the size of the
corpus, and the band remains a query-time choice rather than something baked into bytes on
disk.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import faiss
import numpy as np

from anomalies_scale.covariance_creation import load_covariance, read_corpus, signature_matrix

#: Provenance carried per indexed row, so a neighbour can be traced back to the interval it
#: came from. `depth` is handled separately, since the band logic needs it directly.
#:
#: `n_cells` is the run width in grid cells, which `depth` only records to the
#: nearest power of two - a 3-cell and a 4-cell run both land at depth 1, and telling them
#: apart afterwards is the difference between diagnosing a banding problem and guessing at one.
#: Any column absent from a corpus is simply not recorded, so older corpora still index.
IDENTITY_COLUMNS = ("stream", "lo", "hi", "n_cells")


#: Structures a band sub-index can take. See :func:`make_band_index`.
INDEX_TYPES = ("exact", "ivf", "hnsw")

#: HNSW graph connectivity, and the build- and query-time search widths. ``ef_search`` is the
#: recall knob: raise it and more of the graph is explored per query.
HNSW_M = 32
HNSW_EF_CONSTRUCTION = 200
HNSW_EF_SEARCH = 64

#: ``nlist`` defaults to this times ``sqrt(n)`` - FAISS's usual starting point for IVF.
NLIST_FACTOR = 4

#: FAISS itself warns below roughly this many training points per centroid, and a coarse
#: quantiser trained on fewer is fitting noise. Bands smaller than ``nlist`` times this fall
#: back to exact search rather than being clustered badly.
MIN_POINTS_PER_CENTROID = 39


def make_band_index(vectors, index_type="exact", nlist=None, nprobe=8, hnsw_m=HNSW_M,
                    ef_construction=HNSW_EF_CONSTRUCTION, ef_search=HNSW_EF_SEARCH,
                    random_state=0):
    """Build the index a single depth band is searched with.

    ``'exact'``
        ``IndexFlatL2``: brute force, no training, and the true nearest neighbour every time.

    ``'ivf'``
        ``IndexIVFFlat``: the vectors are clustered into ``nlist`` Voronoi cells by k-means,
        and a query searches only the ``nprobe`` nearest cells.

    ``'hnsw'``
        ``IndexHNSWFlat``: a navigable small-world graph with ``hnsw_m`` links per node, walked
        greedily from an entry point with a candidate list of width ``ef_search``. No training
        and no clustering.

    All three store their vectors in full, so the distances reported are exact. What becomes
    approximate under the latter two is *which* neighbours get found at all.

    Choosing between the approximate two
    ------------------------------------
    IVF earns its speedup from anisotropy - from the vectors falling into separable clumps for
    k-means to find. This index is searched *after* whitening, and ``Sigma^-1/2`` gives the
    corpus unit variance in every retained direction by definition. Whitening therefore removes
    precisely the structure IVF relies on, and measured recall on isotropic data is poor.

    HNSW does not cluster. It navigates a graph built from the vectors' own neighbour
    relations, which survive whitening intact, so it is the better-founded choice here - and
    ``ef_search`` buys recall back continuously, where ``nprobe`` past a point simply
    approaches exhaustive search.

    Returns
    -------
    (faiss.Index, dict)
        The index, and what was actually built - including whether the request fell back.
    """
    if index_type not in INDEX_TYPES:
        raise ValueError("index_type must be one of {0}, got {1!r}".format(
            INDEX_TYPES, index_type))

    vectors = np.ascontiguousarray(vectors, dtype="float32")
    n_rows, dimension = vectors.shape

    def flat(reason=None):
        index = faiss.IndexFlatL2(dimension)
        index.add(vectors)
        info = {"index_type": "exact", "n_vectors": int(n_rows)}
        if reason:
            info.update(fell_back=True, reason=reason)
        return index, info

    if index_type == "exact":
        return flat()

    if index_type == "hnsw":
        # A graph needs more nodes than each node has links, or every node is connected to
        # every other and the walk is an exhaustive scan with pointer chasing on top.
        if n_rows <= 2 * hnsw_m:
            return flat("{0} vector(s) against {1} links per node; the graph would be nearly "
                        "complete and slower than a flat scan".format(n_rows, hnsw_m))

        index = faiss.IndexHNSWFlat(dimension, int(hnsw_m))
        index.hnsw.efConstruction = int(ef_construction)
        index.add(vectors)
        index.hnsw.efSearch = int(ef_search)
        return index, {
            "index_type": "hnsw", "n_vectors": int(n_rows), "M": int(hnsw_m),
            "ef_construction": int(ef_construction), "ef_search": int(ef_search),
        }

    wanted = int(nlist) if nlist else max(1, int(NLIST_FACTOR * np.sqrt(n_rows)))
    affordable = max(1, n_rows // MIN_POINTS_PER_CENTROID)
    chosen = min(wanted, affordable, n_rows)

    # One cell is a flat index with extra steps, and the clustering cost buys nothing.
    if chosen < 2:
        return flat("{0} vector(s) supports fewer than 2 centroids at {1} points each; IVF "
                    "would be a flat index with a clustering step in front of it".format(
                        n_rows, MIN_POINTS_PER_CENTROID))

    quantiser = faiss.IndexFlatL2(dimension)
    index = faiss.IndexIVFFlat(quantiser, dimension, chosen, faiss.METRIC_L2)
    # Deterministic clustering: two runs of the workflow should give one index, not two.
    index.cp.seed = int(random_state)
    index.train(vectors)
    index.add(vectors)
    index.nprobe = int(min(max(nprobe, 1), chosen))

    return index, {
        "index_type": "ivf", "n_vectors": int(n_rows), "nlist": int(chosen),
        "nprobe": int(index.nprobe), "nlist_requested": int(wanted),
        "fraction_searched": float(index.nprobe / chosen),
    }


def _storable(array):
    """Cast an array into something ``npz`` can hold without pickling.

    Stream names arrive from pandas as an object array, and the metadata is deliberately
    loaded with ``allow_pickle=False`` - an index should never be able to execute code on
    being read.
    """
    array = np.asarray(array)
    return array.astype(str) if array.dtype == object else array


def whiten(signatures, covariance):
    """Map signatures into the whitened space the index is searched in.

    ``Sigma^-1/2`` is symmetric, so the transpose is a formality - it is kept to match the
    convention used elsewhere in the package, where the matrix is applied on the right.
    """
    return np.asarray(signatures, dtype=float) @ np.asarray(covariance, dtype=float).T


def whitened_scale(signatures, covariance, sample=4096, random_state=0):
    """Mean squared norm of the corpus once `covariance` has been applied to it.

    This is the defining property of a whitening matrix, and so the one reliable way to tell
    the two forms apart. If ``W = Sigma^-1/2`` then the whitened corpus has unit variance in
    every retained direction, and this mean is exactly the retained rank - at most the
    dimension. If ``Sigma^-1`` is passed instead the whitened covariance is ``Sigma^-1``
    itself, whose trace is the sum of the inverse eigenvalues and therefore enormous on an
    ill-conditioned corpus.

    Sampling keeps it cheap; the quantity is a mean, so a few thousand rows settle it.
    """
    signatures = np.asarray(signatures, dtype=float)
    if signatures.shape[0] > sample:
        rows = np.random.default_rng(random_state).choice(
            signatures.shape[0], size=sample, replace=False)
        signatures = signatures[rows]

    centred = signatures - signatures.mean(axis=0)
    return float(np.mean(np.sum((centred @ np.asarray(covariance, dtype=float).T) ** 2,
                                axis=1)))


def idempotency_error(signatures, covariance, probes=4, sample=4096, random_state=0):
    """How far the whitened corpus covariance is from being a projection.

    This is the test that actually separates the two forms, and it works whatever the
    corpus's conditioning. Write ``C`` for the covariance of the whitened corpus. If the
    matrix applied is ``Sigma^-1/2`` then ``C = Sigma^-1/2 Sigma Sigma^-1/2`` is the identity
    on the retained subspace - a projection, so ``C @ C == C``. If ``Sigma^-1`` is applied
    instead, ``C = Sigma^-1``, which is idempotent only in the degenerate case ``Sigma = I``.

    A magnitude test cannot do this job: on a well-conditioned corpus the two forms are of
    comparable size, and the substitution slips through. Idempotency is a structural property
    and does not care.

    ``C`` is never formed. It is applied to a few random probe vectors through the whitened
    sample, which costs ``O(sample * dimension)`` per probe rather than ``O(dimension ** 3)``.

    Returns
    -------
    float
        Largest relative ``||C(Cu) - Cu|| / ||Cu||`` over the probes. Near zero for a
        whitening; order one or more otherwise.
    """
    signatures = np.asarray(signatures, dtype=float)
    rng = np.random.default_rng(random_state)

    if signatures.shape[0] > sample:
        signatures = signatures[rng.choice(signatures.shape[0], size=sample, replace=False)]

    whitened = (signatures - signatures.mean(axis=0)) @ np.asarray(covariance, dtype=float).T
    n_rows, dimension = whitened.shape

    def apply(vector):
        return whitened.T @ (whitened @ vector) / n_rows

    worst = 0.0
    for _ in range(probes):
        probe = apply(rng.standard_normal(dimension))
        norm = np.linalg.norm(probe)
        if norm <= 1e-12:                     # probe fell outside the retained subspace
            continue
        worst = max(worst, float(np.linalg.norm(apply(probe) - probe) / norm))
    return worst


def check_is_whitening(signatures, covariance, tolerance=1e-4):
    """Raise if `covariance` is not the square root form a flat L2 index needs.

    Checked because the failure is otherwise entirely silent: the index builds, the searches
    run, and the distances look perfectly reasonable while being wrong by orders of magnitude.
    """
    error = idempotency_error(signatures, covariance)
    if error > tolerance:
        raise ValueError(
            "the matrix supplied does not whiten this corpus - it is almost certainly "
            "Sigma^-1 rather than Sigma^-1/2.\n"
            "  Whitening should leave the corpus with a projection for its covariance, but "
            "C @ C differs from C by a relative {0:.3g}.\n"
            "  A flat L2 index measures Euclidean distance, so it needs the square root: "
            "whitening with Sigma^-1 computes (x - y) @ Sigma^-2 @ (x - y).T instead, which "
            "weights every direction quadratically.\n"
            "  Rebuild the covariance with form='inv_sqrt'.".format(error))
    return whitened_scale(signatures, covariance)


def stratified_subsample(depths, cap, random_state=None):
    """Row selection capping the searchable set, sampled proportionally within each depth.

    Stratifying matters: sampling uniformly would thin the shallow depths, which hold only a
    handful of intervals per stream, out of existence long before the deep ones felt it.
    """
    depths = np.asarray(depths)
    if cap is None or cap >= depths.size:
        return np.arange(depths.size)

    rng = np.random.default_rng(random_state)
    fraction = cap / depths.size
    selected = []
    for depth in np.unique(depths):
        rows = np.flatnonzero(depths == depth)
        take = max(1, int(round(fraction * rows.size)))
        selected.append(rng.choice(rows, size=take, replace=False))
    return np.sort(np.concatenate(selected))


def build_index(corpora, covariance, output_index, output_meta=None, band=1,
                nn_reference_size=None, random_state=None, check_whitening=False,
                store_raw=False, show_progress=False):
    """Whiten the given corpora and build one flat L2 index over them.

    Parameters
    ----------
    corpora : mapping
        ``{split_name: corpus}``, each a path or DataFrame. Every split is indexed; the names
        are recorded per row so later stages can tell them apart.
    covariance : str or Path or np.ndarray
        ``Sigma^-1/2``, as written by :mod:`covariance_creation` with ``form='inv_sqrt'``.
    output_index : str or Path
        Where to write the FAISS index.
    output_meta : str or Path, optional
        Where to write the row metadata - depths, splits and interval identity - without
        which the index is unusable for anything but a raw nearest-neighbour lookup.
    band : int
        Depths either side of a query included in its reference set. Recorded, not applied:
        the index holds every row and :class:`PooledIndex` derives the band on load.
    nn_reference_size : int, optional
        Cap on searchable vectors, sampled stratified across depths.
    store_raw : bool
        Also keep the reference vectors as they were *before* whitening. Needed only by the
        off-manifold test, which asks whether a difference lies outside the corpus's row space -
        a question the whitened vectors cannot answer, because ``Sigma^-1/2`` projects onto the
        retained subspace and annihilates exactly the component being looked for. Off by
        default: it is the corpus over again, and on a wide corpus that is hundreds of MB.

    Returns
    -------
    dict
        Diagnostics: rows indexed per split, dimension, and the depths present.
    """
    frames = {name: read_corpus(source) for name, source in corpora.items()}
    if not frames:
        raise ValueError("no corpora given to index")

    # Only the provenance every corpus actually carries is recorded. A column present in one
    # split and missing from another would give ragged arrays that npz cannot hold, so the
    # intersection is taken rather than the union.
    available = [key for key in IDENTITY_COLUMNS
                 if all(key in frames[name].columns for name in frames)]

    columns = None
    blocks, depths, splits, identity = [], [], [], {key: [] for key in available}

    for name in sorted(frames):
        frame = frames[name]
        if "depth" not in frame.columns:
            raise ValueError(
                "corpus {0!r} has no 'depth' column. The index is searched by depth band, so "
                "an interval that cannot say how wide it is cannot be indexed.".format(name))
        signatures, frame_columns = signature_matrix(frame)
        if columns is None:
            columns = frame_columns
        elif frame_columns != columns:
            raise ValueError(
                "corpus {0!r} has different signature columns from the others; they cannot "
                "share one index".format(name))

        blocks.append(signatures)
        depths.append(frame["depth"].to_numpy(dtype=int))
        splits.append(np.full(len(frame), name, dtype=object))
        for key in available:
            identity[key].append(frame[key].to_numpy())

    signatures = np.concatenate(blocks, axis=0)
    depths = np.concatenate(depths)
    splits = np.concatenate(splits)

    if isinstance(covariance, (str, Path)):
        covariance = load_covariance(covariance, expected_columns=columns)
    covariance = np.asarray(covariance, dtype=float)

    # Measured against the fit rows, which are what the metric was fitted on and so the only
    # rows a whitening is guaranteed to bring to unit variance.
    reference = blocks[sorted(frames).index("fit")] if "fit" in frames else signatures

    # Off by default: inside the workflow the covariance comes from the `covariance` rule and
    # is the right form by construction, so paying for four probe multiplies on every build is
    # a check on something the DAG already guarantees.
    #
    # It is kept, rather than deleted, because the one thing the DAG does not guarantee is the
    # config: `metric.form` is a key, and setting it to `pinv` would produce a matrix that
    # indexes and searches perfectly happily while computing Sigma^-2 distances. Turn this on
    # when feeding the module by hand, or after changing that key.
    scale = (check_is_whitening(reference, covariance) if check_whitening
             else whitened_scale(reference, covariance))

    keep = stratified_subsample(depths, nn_reference_size, random_state)
    whitened = np.ascontiguousarray(whiten(signatures[keep], covariance), dtype="float32")
    depths, splits = depths[keep], splits[keep]

    index = faiss.IndexFlatL2(whitened.shape[1])
    index.add(whitened)

    output_index = Path(output_index)
    output_index.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(output_index))

    per_split = {name: int((splits == name).sum()) for name in sorted(frames)}
    info = {
        "n_vectors": int(index.ntotal),
        "dimension": int(whitened.shape[1]),
        "band": int(band),
        "depths": sorted(int(d) for d in np.unique(depths)),
        "per_split": per_split,
        "subsampled": bool(keep.size < signatures.shape[0]),
        # Equals the retained rank when the metric is a genuine whitening; see
        # check_is_whitening. Recorded so a bad metric is visible after the fact too.
        "whitened_scale": scale,
    }

    if output_meta is not None:
        output_meta = Path(output_meta)
        output_meta.parent.mkdir(parents=True, exist_ok=True)
        extra = {}
        if store_raw:
            extra["raw"] = signatures[keep].astype("float32")
            info["raw_stored"] = True
        np.savez_compressed(
            output_meta,
            depth=depths,
            split=splits.astype(str),
            band=np.asarray(band),
            dimension=np.asarray(whitened.shape[1]),
            terms=np.asarray(columns, dtype=str),
            **extra,
            **{key: _storable(np.concatenate(values)[keep])
               for key, values in identity.items()},
        )

    if show_progress:
        print("indexed {n_vectors:,} vectors x {dimension} terms, depths {depths}, band "
              "{band}".format(**info))
        for name, count in per_split.items():
            print("  {0}: {1:,} intervals".format(name, count))
        print("wrote {0}".format(output_index))

    return info


class PooledIndex:
    """A built index, plus the row metadata that makes it queryable by depth.

    Holds one flat index over every reference vector. A query at depth *d* is served by a
    sub-index over the rows within ``band`` of *d*, reconstructed from the flat index on
    first use and cached - exact, and cheap enough that persisting each band separately
    would only cost disk.
    """

    def __init__(self, index, depth, split, band=1, terms=None, identity=None,
                 index_type="exact", nlist=None, nprobe=8, hnsw_m=HNSW_M,
                 ef_construction=HNSW_EF_CONSTRUCTION, ef_search=HNSW_EF_SEARCH,
                 random_state=0, raw=None):
        self.index = index
        self.depth = np.asarray(depth)
        self.split = np.asarray(split)
        #: Reference vectors before whitening, when `build_index` was asked to keep them.
        #: None otherwise, and the off-manifold test then has nothing to compare against.
        self.raw = None if raw is None else np.asarray(raw)
        self.band = int(band)
        self.terms = list(terms) if terms is not None else None
        self.identity = identity or {}
        self.available_depths = np.unique(self.depth)
        # The band structure is chosen here rather than baked into the stored index, because
        # this is where every search actually runs - the stored index is only the vector
        # store. It can therefore be changed on load without rebuilding anything.
        self.index_type = index_type
        self.nlist = nlist
        self.nprobe = nprobe
        self.hnsw_m = hnsw_m
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self.random_state = random_state
        self.band_info = {}
        self._bands = {}

    @property
    def dimension(self):
        return self.index.d

    @property
    def reference_size(self):
        """How many corpus intervals back this detector.

        Named rather than reached for through `.index.ntotal`, so that a detector which is not
        a vector store at all - the isolation forests - can answer the same question.
        """
        return int(self.index.ntotal)

    def vectors(self, rows=None):
        """The whitened reference vectors, or a subset of them.

        A flat index stores its vectors verbatim, so this is a reconstruction in name only -
        it is how the index doubles as the vector store and saves persisting them twice.
        """
        stored = self.index.reconstruct_n(0, self.index.ntotal)
        return stored if rows is None else stored[np.asarray(rows)]

    def band_rows(self, depth):
        """Rows of the reference set within `band` of `depth`."""
        return np.flatnonzero(np.abs(self.depth - depth) <= self.band)

    def band_index(self, depth):
        """Sub-index over `depth`'s band, built on first use."""
        depth = int(depth)
        if depth not in self._bands:
            rows = self.band_rows(depth)
            if rows.size == 0:
                raise ValueError(
                    "no reference intervals within band {0} of depth {1}; the index covers "
                    "depths {2}".format(self.band, depth, list(self.available_depths)))
            vectors = self.index.reconstruct_n(0, self.index.ntotal)[rows]
            sub, info = make_band_index(
                vectors, self.index_type, self.nlist, self.nprobe, self.hnsw_m,
                self.ef_construction, self.ef_search, self.random_state)
            self.band_info[depth] = info
            self._bands[depth] = (sub, rows)
        return self._bands[depth]

    def depth_for_width(self, relative_width):
        """Assign an interval of the given relative width to the closest indexed depth.

        Compared on a log scale, since depth *d* is a relative width of ``2 ** -d``. Answering
        by scale rather than by tree position is what lets intervals that are not dyadic nodes
        at all - those produced by boundary extension - still be scored.
        """
        relative_width = min(max(float(relative_width), 1e-12), 1.0)
        target = -np.log2(relative_width)
        return int(self.available_depths[np.argmin(np.abs(self.available_depths - target))])

    def search(self, queries, depth, k=1, exclude_self=False):
        """Squared Mahalanobis distance from already-whitened `queries` to their neighbours.

        `exclude_self` takes the second neighbour instead of the first, which is what
        calibration needs: a reference vector scored against an index containing it matches
        itself at distance zero, and the second is the one that means anything.
        """
        sub, rows = self.band_index(depth)
        queries = np.ascontiguousarray(np.atleast_2d(queries), dtype="float32")
        wanted = k + 1 if exclude_self else k
        distances, neighbours = sub.search(queries, wanted)
        if exclude_self:
            distances, neighbours = distances[:, 1:], neighbours[:, 1:]
        return distances, rows[neighbours]


def load_index(index_path, meta_path, index_type="exact", nlist=None, nprobe=8,
               hnsw_m=HNSW_M, ef_construction=HNSW_EF_CONSTRUCTION,
               ef_search=HNSW_EF_SEARCH, random_state=0):
    """Read an index and its metadata back into a :class:`PooledIndex`.

    ``index_type`` is a load-time choice, not a property of the file. The stored artifact is a
    flat vector store either way; how each depth band is structured for searching is decided
    here, so switching between exact and IVF costs a reload rather than a rebuild.
    """
    index = faiss.read_index(str(index_path))
    meta = np.load(meta_path, allow_pickle=False)
    identity = {key: meta[key] for key in IDENTITY_COLUMNS if key in meta.files}
    return PooledIndex(
        index,
        depth=meta["depth"],
        split=meta["split"],
        band=int(meta["band"]),
        terms=meta["terms"] if "terms" in meta.files else None,
        identity=identity,
        index_type=index_type,
        nlist=nlist,
        nprobe=nprobe,
        hnsw_m=hnsw_m,
        ef_construction=ef_construction,
        ef_search=ef_search,
        random_state=random_state,
        raw=meta["raw"] if "raw" in meta.files else None,
    )


def main(argv=None):
    """Command line entry point."""
    parser = argparse.ArgumentParser(
        description="Whiten a signature corpus and build its FAISS index.")
    parser.add_argument("--corpus", action="append", dest="corpora", required=True,
                        metavar="SPLIT=PATH",
                        help="corpus to index, as split=path; repeatable")
    parser.add_argument("--covariance", required=True,
                        help="Sigma^-1/2 CSV from covariance_creation (form=inv_sqrt)")
    parser.add_argument("--output", required=True, help="FAISS index to write")
    parser.add_argument("--output-meta", default=None,
                        help="npz of row depths, splits and interval identity")
    parser.add_argument("--band", type=int, default=1,
                        help="depths either side of a query in its reference set")
    parser.add_argument("--nn-reference-size", type=int, default=None,
                        help="cap on searchable vectors, stratified across depths")
    parser.add_argument("--random-state", type=int, default=None, help="seed for that cap")
    parser.add_argument("--quiet", action="store_true", help="suppress progress output")

    args = parser.parse_args(argv)

    corpora = {}
    for entry in args.corpora:
        if "=" not in entry:
            parser.error("--corpus expects split=path, got {0!r}".format(entry))
        name, path = entry.split("=", 1)
        corpora[name] = path

    info = build_index(corpora, args.covariance, args.output, output_meta=args.output_meta,
                       band=args.band, nn_reference_size=args.nn_reference_size,
                       random_state=args.random_state, show_progress=not args.quiet)
    if not args.quiet:
        print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
