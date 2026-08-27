"""Reduce the channel count with an aligned UMAP, before anything is signed.

Why the pressure is on the channel count
----------------------------------------
A truncated signature over ``d`` channels at level ``m`` has ``(d ** (m + 1) - d) / (d - 1)``
terms: polynomial in the length of the path, but *exponential* in the truncation level, with
``d`` as the base of that exponent. SMD's 38 channels become 39 with the time channel adjoined,
and at ``trunc: 4`` that is 2,374,320 terms. Rank truncation downstream cannot help, because
the vectors cannot be formed in the first place. Cutting ``d`` is the only lever that acts on
the exponent - 38 value channels reduced to 4 take those 2.3 million terms to 780.

Why *aligned* UMAP and not one UMAP
-----------------------------------
A signature is only comparable to another signature of a path in the same coordinates. Fit a
separate UMAP per stream and every stream lands in its own arbitrary frame; the Mahalanobis
distance between two such signatures is then meaningless, and the whole detector rests on that
distance. So all streams must end up in one coordinate system.

Plain UMAP fitted on the pooled rows of every stream would also give one coordinate system,
and more cheaply. What :class:`umap.AlignedUMAP` buys instead is one embedding *per stream*,
each free to follow its own local geometry, tied together by an alignment penalty so that
corresponding points stay in corresponding places. That is the right instrument when streams
share structure but not scale - different machines running the same workload, say.

It buys that at the cost of an assumption, and it is a strong one. Alignment is expressed
through ``relations``: a correspondence between the rows of consecutive slices. This module
builds the identity relation, so slice ``i`` row ``t`` is asserted to correspond to slice
``i + 1`` row ``t`` - **that streams are synchronised, and row ``t`` means the same moment in
all of them**. Where that holds, the alignment penalty is doing real work. Where it does not,
it is actively distorting the embedding to force unrelated states together. Check it against
your data before trusting the result, and pass ``relations=`` explicitly if the correspondence
is something other than shared row position.

Where this sits
---------------
Between `preprocessing` and anything that signs: `corpus` in the pooled workflow, `detect` in
the per-interval one. Both take named, screened streams and hand them to ``iisignature``, so
this is the last seam at which the channel count can still change.

The time channel is never reduced. Paths are ``(time, value...)``, and the time coordinate is
what makes a signature see path shape rather than endpoints - it also carries the base-pointing
and the interval slicing. Only the value channels go through UMAP; time is passed through
untouched, so a ``(n, 1 + d)`` path becomes ``(n, 1 + n_components)``.

Scaling is not the normalisation layer that was removed
-------------------------------------------------------
The pipeline dropped its normalisation stage because a Mahalanobis distance accounts for
variance itself, which is correct - downstream of the signature. UMAP is upstream of it, and
builds its neighbour graph under a plain Euclidean metric with no such protection, so a channel
measured in thousands silently decides the embedding while one measured in fractions does not
appear at all. ``standardise=True`` therefore fits a :class:`~sklearn.preprocessing.StandardScaler`
on the fit streams only and persists it inside the projection. This is a different concern at a
different stage, not a reintroduction of the layer that went.

What ``verify`` measures
------------------------
With ``verify=True`` a stratified sample of whole streams is withheld, the alignment is fitted
on the remainder, and the withheld streams are pushed through :meth:`UMAPProjection.transform` -
exactly the path production data will take. Six metrics are computed on them, each with a
threshold in :data:`DEFAULT_THRESHOLDS`:

``trustworthiness``
    Do points that are neighbours in the embedding come from neighbours in the ambient space?
    Penalises *false* neighbours - unrelated states collapsed together, which is the failure
    that would erase an anomaly. Higher is better.
``continuity``
    The converse: are ambient neighbours still neighbours after embedding? Penalises tearing a
    genuine structure apart. Computed as trustworthiness with the two spaces swapped.
``knn_overlap``
    The blunt version of both - what fraction of each point's ``k`` ambient neighbours survive
    as embedded neighbours. Easier to reason about than either, and much less forgiving; 0.3-0.5
    is respectable for UMAP, which is not trying to preserve neighbourhoods exactly.
``step_inflation``
    **The one that matters most here, and the one no standard UMAP score covers.** Signatures
    integrate increments, so consecutive-point distance is not a detail of the embedding, it is
    the input. UMAP optimises a topological objective and is under no obligation to be smooth in
    the ambient space, so adjacent timestamps can land far apart. This measures mean step length
    relative to the cloud's own spread, in the embedding over the same in the ambient space. At
    1.0 the path is exactly as jagged as it was; well above that, the embedding is manufacturing
    path length and the signature will be dominated by spurious jitter.
``generalisation_gap``
    In-sample trustworthiness (streams the alignment was fitted on, using their aligned
    embeddings) minus out-of-sample (withheld streams, through ``transform``). This is the point
    of withholding: it catches an embedding that memorised its fit streams, and it also catches
    the case where ``transform`` is a poorer instrument than the fitted embedding, since those
    are two different code paths and only one of them ships.
``alignment_drift``
    Whether "one coordinate system" is actually true. Each slice has its own mapper; the same
    withheld rows are pushed through several of them and the disagreement is measured against
    the embedding's own diameter. Near zero, the slices agree and a single reference mapper is a
    fair representative of all of them. Large, and the alignment did not take - which invalidates
    the premise, not just the quality.

    Measured against the assumption it is meant to catch, on 20-channel synthetic streams with a
    3-dimensional latent structure. Where the streams were synchronised, so the identity relation
    was true, drift came to 0.098; where each stream carried a phase shift, making row ``t`` a
    different state in every stream, it came to 1.343. The default threshold of 0.25 sits between
    them. Note that trustworthiness was 0.98 in *both* cases - the embedding was locally faithful
    either way, and this is the only metric that noticed the premise had failed.

Three more come from refitting the whole alignment under ``n_seeds`` different random seeds and
comparing the results on common probe rows. UMAP's optimisation is stochastic - random
initialisation, negative sampling, asynchronous SGD - so every run returns a different
embedding. What these ask is whether the *structure* is a property of the data or of the
optimiser: if independent seeds disagree, two runs of the pipeline would disagree too, and any
anomaly score derived from one of them is an accident of that run.

``seed_neighbour_stability``
    Mean pairwise neighbourhood overlap between seeds. Invariant to anything that preserves
    neighbourhoods, so it asks what the detector cares about - do the same points stay together -
    and forgives a global warp.
``seed_procrustes_disparity``
    Mean pairwise residual after optimally rotating, scaling and translating one seed's
    embedding onto another's. Stricter, and necessary here: a Mahalanobis distance is computed
    from coordinates, not from a neighbour list, so two seeds can preserve every neighbourhood
    while disagreeing about shape. Note that comparing coordinates *without* this alignment
    would be meaningless - UMAP fixes its layout only up to rotation, reflection and scale.
``seed_quality_spread``
    Standard deviation of per-seed trustworthiness. Distinguishes two very different situations
    that the first two cannot: seeds that disagree with each other while each being an equally
    good representation, and seeds that vary in quality. Only the latter makes the verdict
    itself depend on which seed you happened to run.

Only the UMAP seed varies across those refits - the fit/withheld split, the row selection and
the scaler are all held fixed, so what is measured is the optimiser's own variability rather
than that plus resampling noise. The projection returned is always the one fitted under
``random_state``; the extra seeds are diagnostic and are discarded.

Calibrated against the contrast they exist to draw - a corpus with a genuine 3-dimensional
latent structure, against one of pure noise where nothing is there to find and whatever UMAP
returns is its own invention:

===========================  ==========  ===========  =========
metric                       structured  noise        threshold
===========================  ==========  ===========  =========
seed_neighbour_stability     0.877       0.285        >= 0.40
seed_procrustes_disparity    0.089       0.584        <= 0.25
seed_quality_spread          0.0006      0.0068       <= 0.05
===========================  ==========  ===========  =========

The first two thresholds sit between the two cases and the noise corpus fails both. The third
does not discriminate here at all - both figures are an order of magnitude inside it - so treat
``seed_quality_spread`` as uncalibrated. It answers a different question from the other two
(whether seeds vary in *quality*, rather than in what they produce) and it did not catch the
one bad corpus tested, so it should not be relied on alone.

A warning about this particular check, because it failed silently once already and the failure
was invisible: before ``fit_aligned`` was corrected to vary ``transform_seed``, all five seeds
returned bit-identical embeddings and the metrics read 1.0000, 0.0000 and 0.0000 - a perfect
score that meant only that the same embedding had been compared with copies of itself. A
verification that fails towards reassurance is worse than none, so :func:`seed_stability` now
detects identical embeddings and withdraws the scores instead of reporting them.

One more test exists that this module deliberately does not run, because it belongs downstream:
whether anomalies survive. UMAP is fitted to normal data and its whole purpose is to represent
that data well; an anomaly is off that manifold, which is exactly where the transform is least
constrained. It may map anomalies back onto normal-looking coordinates - erasing the signal the
pipeline exists to find - or throw them far away, which helps. No unsupervised metric here can
tell you which. Run the detector on data with known anomalies, with and without the reduction,
and compare.

A failed verification is reported, not enforced. :func:`fit_projection` returns the learnt
projection either way; ``projection.is_valid`` and ``projection.report`` say what was found.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from anomalies_scale.canonical_streams import iter_streams

#: Value channels the reduction targets by default.
DEFAULT_N_COMPONENTS = 4

#: Instants on each side that each embedding is aligned against - ``AlignedUMAP``'s own
#: default, restated here so it is a decision rather than an inherited one. This is the knob
#: that governs how rigidly the latent trajectory holds together over time, and therefore how
#: smooth the paths a signature will integrate come out.
DEFAULT_ALIGNMENT_WINDOW = 3

#: How the reduced channels are named once a projected stream is written back out.
COMPONENT_TEMPLATE = "Component_{0}"

#: Neighbourhood size used by the quality metrics. Independent of UMAP's own ``n_neighbors``:
#: this asks what the embedding preserved, and reusing the number the fit was told to optimise
#: would be marking its own homework.
DEFAULT_METRIC_NEIGHBOURS = 15

#: Rows sampled per stream for the O(n^2) metrics. Trustworthiness and continuity build a full
#: pairwise distance matrix, so an SMD stream of 28,479 rows would need ~6 GB per stream.
DEFAULT_METRIC_SAMPLE = 2000

#: Independent refits used to measure seed stability. Each costs a full fit, so this multiplies
#: the price of verification - but an embedding that moves when only the seed moves is an
#: artefact of the optimiser, and nothing downstream of it means anything.
DEFAULT_N_SEEDS = 5

#: Pass marks. Each is a judgement call, not a law - override with ``thresholds=``.
DEFAULT_THRESHOLDS = {
    "trustworthiness": 0.85,
    "continuity": 0.85,
    "knn_overlap": 0.30,
    "step_inflation": 2.0,
    "generalisation_gap": 0.05,
    "alignment_drift": 0.25,
    "seed_neighbour_stability": 0.40,
    "seed_procrustes_disparity": 0.25,
    "seed_quality_spread": 0.05,
}

#: Which way each metric wants to go: ``ge`` passes at or above the threshold, ``le`` at or
#: below it.
METRIC_DIRECTION = {
    "trustworthiness": "ge",
    "continuity": "ge",
    "knn_overlap": "ge",
    "step_inflation": "le",
    "generalisation_gap": "le",
    "alignment_drift": "le",
    "seed_neighbour_stability": "ge",
    "seed_procrustes_disparity": "le",
    "seed_quality_spread": "le",
}


def import_umap():
    """Import :mod:`umap` on demand, with an error that says what to install.

    Deferred rather than imported at module scope for two reasons. ``umap-learn`` pulls in
    numba and llvmlite, which pin numpy hard and take several seconds to warm up, and the rest
    of the pipeline runs perfectly well without any of it - the reduction is optional. Nothing
    should pay that cost, or inherit that dependency conflict, for importing a module it is
    not going to use.
    """
    try:
        import umap
    except ImportError as error:  # pragma: no cover - depends on the environment
        raise ImportError(
            "the UMAP reduction needs umap-learn, which is not installed. "
            "`pip install umap-learn` - note that it pulls in numba and llvmlite, which pin "
            "numpy versions tightly, so install it into this project's environment and check "
            "the rest of the pipeline still imports afterwards."
        ) from error
    return umap


# ---------------------------------------------------------------------------------------
# Loading


def load_corpus_streams(files, stream_col=None, time_col=None, value_cols=None):
    """Read every stream out of ``files`` as ``(name, path_array)`` pairs.

    Shares :func:`~anomalies_scale.canonical_streams.iter_streams` with the corpus builder,
    so a file holding many streams is split the same way here as it is there, and a stream
    means the same thing to both.
    """
    streams = []
    for file in files:
        for name, path in iter_streams(file, stream_col, time_col, value_cols):
            streams.append((name, np.asarray(path, dtype=float)))
    if not streams:
        raise ValueError("no streams found in {0}".format([str(f) for f in files]))
    return streams


def check_channel_widths(streams):
    """Every stream must carry the same channels, and return how many value channels that is.

    A stream with a different width is not a stream this projection can be fitted across - the
    slices would not share a feature space, and the alignment relation would be relating
    columns that do not correspond.
    """
    widths = {path.shape[1] for _, path in streams}
    if len(widths) != 1:
        summary = ", ".join(
            "{0}: {1}".format(name, path.shape[1]) for name, path in streams[:8])
        raise ValueError(
            "streams carry differing channel counts ({0}); all must share one feature "
            "space. First few - {1}".format(sorted(widths), summary))
    width = widths.pop()
    if width < 2:
        raise ValueError(
            "streams have {0} column(s); expected time plus at least one value "
            "channel".format(width))
    return width - 1


def shared_row_selection(streams, max_rows, random_state):
    """Row indices used to fit, shared by every slice.

    Two jobs at once. Slices must be index-comparable for the identity relation to mean
    anything, so the selection has to be the *same* rows in every stream - which also caps the
    fit at the shortest stream. And an aligned fit over 28 SMD streams of 28,479 rows is hours
    of work for an embedding that a few thousand rows determines just as well, so ``max_rows``
    thins it.

    Rows are drawn without replacement and returned in order, so each slice stays a path
    rather than a shuffled cloud. Thinning a path is fine for the fit - UMAP has no notion of
    row order - and every row is still transformable afterwards.
    """
    lengths = [len(path) for _, path in streams]
    shortest = min(lengths)
    if max_rows is None or max_rows >= shortest:
        return np.arange(shortest), shortest < max(lengths)
    rng = np.random.default_rng(random_state)
    return np.sort(rng.choice(shortest, size=max_rows, replace=False)), True


# ---------------------------------------------------------------------------------------
# Fitting


def identity_relations(n_slices, n_rows):
    """Identity correspondences between consecutive slices.

    ``umap.AlignedUMAP`` wants one dictionary per consecutive pair, mapping a row of one slice
    to the row it corresponds to in the next. Every slice here holds the same row selection, so
    the correspondence is the identity - and being symmetric, it sidesteps the question of which
    direction the convention runs in.

    This is the synchronisation assumption made concrete: it asserts row ``t`` of one stream is
    the counterpart of row ``t`` of the next. See the module docstring.
    """
    return [{i: i for i in range(n_rows)} for _ in range(n_slices - 1)]


def adopt_aligned_embeddings(aligned):
    """Give each per-slice mapper the aligned embedding in place of its own.

    ``AlignedUMAP`` fits an independent ``UMAP`` per slice and *then* runs a joint optimisation,
    leaving the aligned coordinates in ``aligned.embeddings_`` while each ``mappers_[i]`` still
    holds the unaligned embedding it arrived with. That split matters, because ``AlignedUMAP``
    has no ``transform``: out-of-sample points can only be placed by a per-slice mapper, and
    such a mapper positions new points relative to its own ``embedding_``. Left alone, it would
    hand back coordinates in the *unaligned* frame - a different space from the one the corpus
    ended up in, which is precisely the failure this module exists to avoid.

    So the aligned coordinates are written back over each mapper's own. After this,
    ``mapper.transform`` places new rows against aligned neighbours and returns aligned
    coordinates.

    The alternative ``AlignedUMAP`` does offer is ``update()``, which admits the new data as a
    further slice - but that refits, letting test data influence the geometry it is about to be
    judged against. That is the same leak the fit/withheld/test discipline exists to prevent, so
    it is not used.

    NaN rows
    --------
    ``AlignedUMAP.fit`` finishes by writing ``NaN`` into every row whose slice graph had no
    edges, marking vertices the neighbour graph left disconnected. Adopted as-is those rows
    would poison the whole projection rather than just themselves: ``UMAP.transform`` places a
    new point at a weighted average of its neighbours' embedded positions, and it chooses those
    neighbours by ambient distance against ``_raw_data``, which knows nothing about graph
    connectivity. One NaN neighbour is enough to return NaN, so a handful of disconnected
    corpus rows would silently turn arbitrary test streams into NaN signatures.

    They are therefore imputed to the centroid of the finite rows - a position that says
    nothing, which is the honest thing for a point the embedding failed to place. The count is
    returned so it can be reported rather than absorbed: a large one means ``n_neighbors`` is
    too small for this data, and the embedding should be refitted, not patched.

    Numba containers
    ----------------
    ``embeddings_`` comes back as a ``numba.typed.List``, which holds a ``_nrt_python._MemInfo``
    and cannot be pickled - so a projection carrying one is unsaveable, and the failure only
    appears at the end of a fit that may have taken hours. It is replaced here with an ordinary
    list of numpy arrays, which is what the rest of this module reads anyway.
    """
    imputed = 0
    adopted = []
    for mapper, embedding in zip(aligned.mappers_, aligned.embeddings_):
        embedding = np.array(embedding, dtype=np.float32, copy=True)
        missing = ~np.isfinite(embedding).all(axis=1)
        if missing.any():
            finite = embedding[~missing]
            if not len(finite):
                raise ValueError(
                    "a slice embedded to entirely non-finite coordinates; its neighbour graph "
                    "was wholly disconnected, so n_neighbors is far too small for this data")
            embedding[missing] = finite.mean(axis=0)
            imputed += int(missing.sum())
        mapper.embedding_ = embedding
        adopted.append(embedding)

    aligned.embeddings_ = adopted
    return aligned, imputed


def fit_aligned(slices, n_components, n_neighbors, min_dist, random_state,
                relations=None, verbose=False, transform_seed=None, **umap_kwargs):
    """Fit an :class:`umap.AlignedUMAP` over ``slices`` and return it, ready to transform.

    ``slices`` is a list of ``(n_rows, n_channels)`` arrays, one per stream, all the same shape.

    Seeding
    -------
    ``random_state`` alone does **not** determine an ``AlignedUMAP`` result, which is worth
    stating because it is the opposite of how every other estimator here behaves. The per-slice
    UMAPs do take it, but the joint alignment then overwrites their embeddings, and that
    optimisation seeds itself from ``transform_seed`` instead - see ``AlignedUMAP.fit``, where
    the seed triplet handed to ``optimize_layout_aligned_euclidean`` comes from
    ``np.random.RandomState(self.transform_seed)``. Measured on 3 slices of 300 rows: varying
    ``random_state`` alone changed the aligned embedding by 0.0 in every coordinate, while
    varying ``transform_seed`` with it changed it by up to 9.05. Plain ``UMAP`` responds to
    ``random_state`` normally, so this is specific to the aligned variant.

    Left alone that would make ``random_state`` inert here: two runs configured differently
    would return the identical projection, and a stability sweep over seeds would compare five
    copies of one embedding and pronounce it perfectly stable. So ``transform_seed`` defaults to
    ``random_state``, which puts both under one knob. ``transform_seed`` also drives
    ``UMAP.transform``'s own stochastic placement, so tying them makes out-of-sample embedding
    reproducible from the same single seed.
    """
    umap = import_umap()

    n_rows = slices[0].shape[0]
    if relations is None:
        relations = identity_relations(len(slices), n_rows)
    if len(relations) != len(slices) - 1:
        raise ValueError(
            "need one relation per consecutive pair: {0} slice(s) require {1} relation(s), "
            "got {2}".format(len(slices), len(slices) - 1, len(relations)))

    # UMAP silently misbehaves when asked for more neighbours than there are points; cap it
    # rather than letting the fit produce something meaningless.
    n_neighbors = int(min(n_neighbors, max(2, n_rows - 1)))

    if transform_seed is None and random_state is not None:
        transform_seed = random_state

    aligned = umap.AlignedUMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=random_state,
        transform_seed=42 if transform_seed is None else transform_seed,
        verbose=verbose,
        **umap_kwargs,
    )
    aligned.fit(slices, relations=relations)
    return adopt_aligned_embeddings(aligned)


def check_transform_is_finite(aligned, rows):
    """Confirm the adopted embeddings actually transform, before anything depends on them.

    Cheap insurance against the NaN path described in :func:`adopt_aligned_embeddings`, and
    against any future change to how ``AlignedUMAP`` marks points it could not place. Failing
    here costs one transform; failing later costs a corpus of NaN signatures and a covariance
    that is NaN throughout.
    """
    probe = np.asarray(aligned.mappers_[0].transform(rows[:min(len(rows), 64)]), dtype=float)
    if not np.isfinite(probe).all():
        raise ValueError(
            "the fitted projection transforms known-good rows to non-finite coordinates; "
            "signatures built from it would be NaN throughout. Raise n_neighbors, or check "
            "the corpus for degenerate streams")


# ---------------------------------------------------------------------------------------
# Quality metrics


def trustworthiness_score(ambient, embedded, n_neighbours):
    """Fraction of embedded neighbourhoods that were neighbourhoods to begin with.

    Venna & Kaski's measure, via scikit-learn. Penalises points pulled together that were far
    apart - the failure mode that would let an anomalous state land among normal ones.
    """
    from sklearn.manifold import trustworthiness

    k = int(min(n_neighbours, (len(ambient) - 1) // 2))
    if k < 1:
        return float("nan")
    return float(trustworthiness(ambient, embedded, n_neighbors=k))


def continuity_score(ambient, embedded, n_neighbours):
    """Trustworthiness with the spaces swapped: the penalty for tearing structure apart.

    Trustworthiness asks whether embedded neighbours were ambient neighbours; continuity asks
    whether ambient neighbours stayed neighbours. They fail independently, and an embedding can
    score well on one while destroying the other, so both are reported.
    """
    return trustworthiness_score(embedded, ambient, n_neighbours)


def knn_overlap(ambient, embedded, n_neighbours):
    """Mean fraction of each point's ``k`` ambient neighbours that remain neighbours embedded.

    Cruder than trustworthiness and much less forgiving - it takes no account of *how* near a
    lost neighbour ended up - but it is the one number here that means something without
    knowing the metric's definition.
    """
    from sklearn.neighbors import NearestNeighbors

    k = int(min(n_neighbours, len(ambient) - 1))
    if k < 1:
        return float("nan")

    # X=None asks for each training point's neighbours excluding itself, which is what the
    # overlap should be computed over.
    ambient_nn = NearestNeighbors(n_neighbors=k).fit(ambient).kneighbors(return_distance=False)
    embedded_nn = NearestNeighbors(n_neighbors=k).fit(embedded).kneighbors(return_distance=False)

    shared = [len(set(a) & set(b)) for a, b in zip(ambient_nn, embedded_nn)]
    return float(np.mean(shared) / k)


def jaggedness(points):
    """Mean step between consecutive rows, relative to the cloud's own spread.

    Scale-free by construction, so the ambient and embedded figures are comparable even though
    the two spaces have no common unit.
    """
    if len(points) < 2:
        return float("nan")
    steps = np.linalg.norm(np.diff(points, axis=0), axis=1)
    spread = np.linalg.norm(points - points.mean(axis=0), axis=1).mean()
    if spread <= 0:
        return float("nan")
    return float(steps.mean() / spread)


def step_inflation(ambient, embedded):
    """How much more jagged the path became - the metric that decides whether this is usable.

    A signature is an iterated integral of increments, so consecutive-point distance is the raw
    material rather than an incidental property. UMAP optimises a topological objective and is
    not obliged to be smooth in the ambient space; where it is not, the embedded path acquires
    length that was never in the data, and a truncated signature - which is dominated by its
    low levels, and so by total variation - reports that manufactured jitter as structure.

    1.0 means the path is exactly as jagged as it was. Rows must be in path order and
    contiguous, so this is computed on whole streams, never on a subsample.
    """
    ambient_jag = jaggedness(ambient)
    embedded_jag = jaggedness(embedded)
    if not np.isfinite(ambient_jag) or ambient_jag <= 0:
        return float("nan")
    return float(embedded_jag / ambient_jag)


def alignment_drift(mappers, rows, reference, probes=4, random_state=0):
    """Disagreement between slice mappers on the same rows, relative to embedding diameter.

    Tests the premise rather than the quality. The whole justification for
    :class:`umap.AlignedUMAP` here is that every slice ends up in one coordinate system, and
    :meth:`UMAPProjection.transform` acts on that by placing all future data through a single
    reference mapper. If the slices in fact disagree about where a point goes, that reference is
    an arbitrary choice among several incompatible frames and the corpus is not in one space at
    all.

    Near zero means the alignment took. Order 1 means it did not.
    """
    if len(mappers) < 2:
        return float("nan")

    baseline = np.asarray(mappers[reference].transform(rows), dtype=float)
    diameter = np.linalg.norm(baseline - baseline.mean(axis=0), axis=1).mean()
    if diameter <= 0:
        return float("nan")

    others = [i for i in range(len(mappers)) if i != reference]
    rng = np.random.default_rng(random_state)
    if len(others) > probes:
        others = sorted(rng.choice(others, size=probes, replace=False).tolist())

    displacements = [
        np.linalg.norm(np.asarray(mappers[i].transform(rows), dtype=float) - baseline,
                       axis=1).mean()
        for i in others
    ]
    return float(np.mean(displacements) / diameter)


def procrustes_disparity(first, second):
    """Residual disagreement between two embeddings after optimal rigid alignment.

    UMAP's layout is only determined up to translation, rotation, reflection and overall scale -
    two runs can produce the same shape in different orientations, and comparing coordinates
    directly would report a difference that means nothing. Procrustes quotients all four out:
    both clouds are centred, scaled to unit norm and optimally rotated onto each other, and what
    is left is genuine disagreement about shape.

    Returns scipy's disparity, the residual sum of squares after that alignment. 0 is identical.
    """
    from scipy.spatial import procrustes

    if first.shape != second.shape or len(first) < 2:
        return float("nan")
    try:
        return float(procrustes(first, second)[2])
    except ValueError:
        # Raised when an input is degenerate - a constant embedding has no shape to align.
        return float("nan")


def seed_stability(slices, probe_values, n_components, n_neighbors, min_dist,
                   random_state, n_seeds=DEFAULT_N_SEEDS, relations=None,
                   n_neighbours=DEFAULT_METRIC_NEIGHBOURS, reference_embedding=None,
                   show_progress=False, **umap_kwargs):
    """Refit the alignment under several seeds and measure whether they agree.

    UMAP's optimisation is stochastic - random initialisation, negative sampling, and an
    asynchronous SGD whose update order is not deterministic. Every run therefore returns a
    different embedding, and the question this answers is whether *the structure* is a property
    of the data or of the optimiser. If independent seeds disagree, then so would two runs of
    the pipeline, and any anomaly score derived from one of them is an accident of that run.

    Each seed's embedding is compared on the same probe rows, which are held out of every fit,
    and compared through :meth:`transform` - the path production data takes - rather than
    through the fitted embeddings, so this measures the stability of what actually ships.

    Only the UMAP seed varies. The fit/withheld split, the row selection and the scaler are all
    held fixed, so what comes back is the optimiser's own variability rather than that plus
    resampling noise.

    Two complementary comparisons, because they fail differently:

    ``seed_neighbour_stability``
        Mean pairwise ``knn_overlap`` between seeds. Invariant to any transformation that
        preserves neighbourhoods, so it asks the question the detector cares about - do the same
        points stay together - and forgives a global warp that a rigid alignment could not.
    ``seed_procrustes_disparity``
        Mean pairwise :func:`procrustes_disparity`. Stricter: it demands the *shape* agree, not
        just the neighbourhoods. Two seeds can score well on overlap while placing a cluster on
        opposite sides of the cloud, which matters here because a Mahalanobis distance is
        computed from coordinates, not from a neighbour list.

    Returns
    -------
    dict
        The two metrics, ``seed_quality_spread`` (standard deviation of per-seed
        trustworthiness), and the per-seed values behind them.
    """
    embeddings = []
    if reference_embedding is not None:
        embeddings.append(np.asarray(reference_embedding, dtype=float))

    for offset in range(len(embeddings), n_seeds):
        if show_progress:
            print("  stability refit {0}/{1} (seed {2})".format(
                offset + 1, n_seeds, random_state + offset))
        aligned, _ = fit_aligned(slices, n_components, n_neighbors, min_dist,
                                 random_state + offset, relations=relations, **umap_kwargs)
        embeddings.append(np.asarray(aligned.mappers_[0].transform(probe_values), dtype=float))

    if len(embeddings) < 2:
        return {"seed_neighbour_stability": float("nan"),
                "seed_procrustes_disparity": float("nan"),
                "seed_quality_spread": float("nan"),
                "seed_trustworthiness": [], "n_seeds": len(embeddings)}

    # If every seed returns bit-identical coordinates, the seed is not reaching the optimiser and
    # this whole check is measuring one embedding against copies of itself - which would report
    # as perfect stability, the most misleading possible answer. That is not hypothetical: it is
    # what `AlignedUMAP` does when only `random_state` varies (see `fit_aligned`). Detected and
    # reported rather than trusted, so a future change upstream cannot quietly revive it.
    identical = all(np.array_equal(embeddings[0], other) for other in embeddings[1:])

    overlaps, disparities = [], []
    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            overlaps.append(knn_overlap(embeddings[i], embeddings[j], n_neighbours))
            disparities.append(procrustes_disparity(embeddings[i], embeddings[j]))

    # Per-seed quality against the ambient probe rows: even when seeds disagree with each other,
    # they may each be equally good representations, and that is a different situation from all
    # of them being bad. The spread is what would make a verdict seed-dependent.
    per_seed = [trustworthiness_score(probe_values, embedding, n_neighbours)
                for embedding in embeddings]

    return {
        "seed_neighbour_stability": float(np.nanmean(overlaps)),
        "seed_procrustes_disparity": float(np.nanmean(disparities)),
        "seed_quality_spread": float(np.nanstd(per_seed)),
        "seed_trustworthiness": [float(v) for v in per_seed],
        "n_seeds": len(embeddings),
        "seeds_identical": bool(identical),
    }


def subsample_rows(array, limit, random_state):
    """Thin an array to ``limit`` rows for the quadratic metrics, preserving order."""
    if limit is None or len(array) <= limit:
        return np.arange(len(array))
    rng = np.random.default_rng(random_state)
    return np.sort(rng.choice(len(array), size=limit, replace=False))


# ---------------------------------------------------------------------------------------
# Verification


class VerificationReport:
    """Metric values, the thresholds they were judged against, and the resulting verdict."""

    def __init__(self, metrics, thresholds, n_fit_streams, n_withheld_streams, notes=()):
        self.metrics = dict(metrics)
        self.thresholds = dict(thresholds)
        self.n_fit_streams = int(n_fit_streams)
        self.n_withheld_streams = int(n_withheld_streams)
        self.notes = list(notes)

    def outcome(self, name):
        """``True``/``False`` for a judged metric, or ``None`` when it could not be computed.

        A metric that came back non-finite is not a pass and not a failure - there were too few
        rows, or a stream stood still. Reporting it as either would be a guess.
        """
        value = self.metrics.get(name)
        threshold = self.thresholds.get(name)
        if value is None or threshold is None or not np.isfinite(value):
            return None
        if METRIC_DIRECTION[name] == "ge":
            return bool(value >= threshold)
        return bool(value <= threshold)

    @property
    def outcomes(self):
        return {name: self.outcome(name) for name in self.thresholds}

    @property
    def is_valid(self):
        """Every judged metric passed, and at least one could be judged.

        An uncomputable metric does not fail the projection, but a report in which *nothing*
        could be computed is not evidence of anything and must not read as a pass.
        """
        outcomes = [o for o in self.outcomes.values() if o is not None]
        return bool(outcomes) and all(outcomes)

    def to_dict(self):
        return {
            "is_valid": self.is_valid,
            "metrics": self.metrics,
            "thresholds": self.thresholds,
            "outcomes": self.outcomes,
            "n_fit_streams": self.n_fit_streams,
            "n_withheld_streams": self.n_withheld_streams,
            "notes": self.notes,
        }

    def summary(self):
        """A readable verdict, one line per metric.

        An unmeasured metric is named in the headline rather than left to the per-metric lines.
        ``is_valid`` means "everything measured passed", and a bare SUITABLE over a report with
        three skipped checks reads as a stronger claim than the evidence supports.
        """
        verdict = "SUITABLE" if self.is_valid else "NOT SUITABLE"
        unmeasured = sum(1 for outcome in self.outcomes.values() if outcome is None)
        if unmeasured:
            verdict += " (on {0} of {1} checks; {2} not measured)".format(
                len(self.thresholds) - unmeasured, len(self.thresholds), unmeasured)
        lines = [
            "UMAP verification: {0}".format(verdict),
            "  fitted on {0} stream(s), verified against {1} withheld".format(
                self.n_fit_streams, self.n_withheld_streams),
        ]
        for name in sorted(self.thresholds):
            outcome = self.outcome(name)
            mark = {True: "pass", False: "FAIL", None: "n/a "}[outcome]
            value = self.metrics.get(name, float("nan"))
            lines.append("  {0}  {1:<19} {2:8.4f}  ({3} {4})".format(
                mark, name, value,
                "needs >=" if METRIC_DIRECTION[name] == "ge" else "needs <=",
                self.thresholds[name]))
        lines.extend("  note: {0}".format(note) for note in self.notes)
        return "\n".join(lines)

    def __repr__(self):
        return "<VerificationReport {0}>".format(
            "valid" if self.is_valid else "invalid")


def stratified_withhold(streams, fraction, random_state):
    """Split streams into fit and withheld, taking whole streams and never parts of one.

    Withholding rows rather than streams would leak: neighbouring rows of one series are close
    to each other, so a row held out while its neighbours are fitted is not unseen data in any
    useful sense, and the metrics would report an optimism that production never sees.

    Stratified by stream length, since length is the one property known before the fit that
    plainly bears on embedding quality; sampling within length bands keeps a corpus of mixed
    lengths from withholding only its short streams.
    """
    n = len(streams)
    n_withheld = max(1, int(round(n * fraction)))
    if n_withheld >= n:
        raise ValueError(
            "verify_fraction {0} would withhold {1} of {2} stream(s), leaving nothing to fit "
            "on".format(fraction, n_withheld, n))

    rng = np.random.default_rng(random_state)
    order = np.argsort([len(path) for _, path in streams], kind="stable")
    bands = np.array_split(order, n_withheld)

    withheld = []
    for band in bands:
        if len(band):
            withheld.append(int(rng.choice(band)))
    withheld = set(withheld)

    fit_idx = [i for i in range(n) if i not in withheld]
    return fit_idx, sorted(withheld)


def verify_projection(projection, withheld, fit_slices, fit_embeddings, thresholds=None,
                      n_neighbours=DEFAULT_METRIC_NEIGHBOURS,
                      metric_sample=DEFAULT_METRIC_SAMPLE, stability=None,
                      drift_probes=4, random_state=0, show_progress=False):
    """Measure the projection against streams it never saw, and return a report.

    ``withheld`` is a list of ``(name, path_array)``. ``fit_slices`` and ``fit_embeddings`` are
    the scaled fit rows and their aligned coordinates, one array per slice, used only for the
    in-sample half of ``generalisation_gap``.

    The in-sample figure is computed **per slice and then averaged**, exactly as the
    out-of-sample one is computed per withheld stream and averaged. Pooling the fit slices into
    one cloud instead would compare different things: a pooled trustworthiness also charges for
    how the slices sit relative to *each other*, which is alignment quality rather than
    embedding quality, and is already measured by ``alignment_drift``. Getting this wrong makes
    the gap negative, which is how it was caught.
    """
    thresholds = dict(DEFAULT_THRESHOLDS if thresholds is None else thresholds)
    notes = []

    per_stream = {"trustworthiness": [], "continuity": [], "knn_overlap": [],
                  "step_inflation": []}

    for position, (name, path) in enumerate(withheld):
        values = projection.scale(path[:, 1:])
        embedded = np.asarray(projection.mapper.transform(values), dtype=float)

        # step_inflation needs contiguous rows in path order, so it sees the whole stream.
        per_stream["step_inflation"].append(step_inflation(values, embedded))

        # The rest are quadratic in the row count and indifferent to order.
        rows = subsample_rows(values, metric_sample, random_state + position)
        ambient_s, embedded_s = values[rows], embedded[rows]
        per_stream["trustworthiness"].append(
            trustworthiness_score(ambient_s, embedded_s, n_neighbours))
        per_stream["continuity"].append(
            continuity_score(ambient_s, embedded_s, n_neighbours))
        per_stream["knn_overlap"].append(knn_overlap(ambient_s, embedded_s, n_neighbours))

        if show_progress:
            print("  verified {0}/{1}: {2}".format(position + 1, len(withheld), name))

    metrics = {key: float(np.nanmean(values)) if values else float("nan")
               for key, values in per_stream.items()}

    # In-sample trustworthiness deliberately uses the aligned embeddings rather than pushing
    # the fit rows back through transform. The gap between the two is exactly what is being
    # measured: the fitted embedding is the quality UMAP achieved, transform is the quality
    # that ships, and they are different code paths.
    #
    # Capped at the number of withheld streams so the two halves of the gap average over
    # comparable sample sizes, and so a 300-slice corpus does not spend its whole verification
    # budget on the in-sample half.
    in_sample = []
    for position, (slice_rows, embedding) in enumerate(
            zip(fit_slices[:max(1, len(withheld))], fit_embeddings)):
        rows = subsample_rows(slice_rows, metric_sample, random_state + position)
        in_sample.append(trustworthiness_score(
            slice_rows[rows], np.asarray(embedding, dtype=float)[rows], n_neighbours))
    in_sample = float(np.nanmean(in_sample)) if in_sample else float("nan")

    metrics["generalisation_gap"] = float(in_sample - metrics["trustworthiness"])
    metrics["in_sample_trustworthiness"] = in_sample

    # Probe rows: a withheld stream, thinned. Held out of every fit including the stability
    # refits, so no seed has an advantage on them, and small enough that five extra fits and
    # their pairwise comparisons stay affordable.
    probe = withheld[0][1][:, 1:]
    probe = projection.scale(probe[subsample_rows(probe, min(500, metric_sample), random_state)])

    metrics["alignment_drift"] = alignment_drift(
        projection.mappers, probe, projection.reference, drift_probes, random_state)

    if stability is not None:
        for key in ("seed_neighbour_stability", "seed_procrustes_disparity",
                    "seed_quality_spread"):
            metrics[key] = stability[key]
        per_seed = stability.get("seed_trustworthiness") or []
        if per_seed:
            notes.append("per-seed trustworthiness over {0} seed(s): {1}".format(
                stability["n_seeds"], ", ".join("{0:.3f}".format(v) for v in per_seed)))
        if stability["n_seeds"] < 2:
            notes.append("seed stability needs at least 2 seeds; it was not measured")
        if stability.get("seeds_identical"):
            # Perfect scores that mean nothing: withdraw them rather than report them.
            for key in ("seed_neighbour_stability", "seed_procrustes_disparity",
                        "seed_quality_spread"):
                metrics[key] = float("nan")
            notes.append(
                "every seed returned a bit-identical embedding, so the seed is not reaching "
                "the optimiser and stability is UNMEASURED, not perfect; the scores have been "
                "withdrawn rather than reported")
    else:
        notes.append("seed stability was not measured (n_seeds=1): the embedding has not been "
                     "shown to be a property of the data rather than of the optimiser")

    if len(withheld) < 3:
        notes.append(
            "only {0} withheld stream(s): every figure here is an average over very few "
            "samples and should be read as indicative".format(len(withheld)))
    if not np.isfinite(metrics["alignment_drift"]):
        notes.append("alignment_drift could not be computed (one slice, or a degenerate "
                     "embedding); the single-coordinate-system premise is untested")

    return VerificationReport(metrics, thresholds, len(fit_slices), len(withheld), notes)


# ---------------------------------------------------------------------------------------
# The projection


class UMAPProjection:
    """A fitted reduction, and everything needed to apply it to a stream it has never seen.

    Holds the aligned mappers, the scaler fitted alongside them, the channel count it expects,
    and the verification report if one was asked for. :meth:`transform` is the only thing the
    rest of the pipeline needs.
    """

    def __init__(self, aligned, scaler, n_value_channels, n_components, reference=0,
                 report=None, stream_names=(), params=None):
        self.aligned = aligned
        self.scaler = scaler
        self.n_value_channels = int(n_value_channels)
        self.n_components = int(n_components)
        self.reference = int(reference)
        self.report = report
        self.stream_names = list(stream_names)
        self.params = dict(params or {})

    @property
    def mappers(self):
        """The per-slice mappers, each carrying the aligned embedding."""
        return self.aligned.mappers_

    @property
    def mapper(self):
        """The slice mapper new data is placed through.

        Any of them would do if the alignment worked perfectly, which is what
        ``alignment_drift`` checks. One is chosen and fixed so that two runs of the pipeline
        cannot silently embed into different frames.
        """
        return self.mappers[self.reference]

    @property
    def is_valid(self):
        """Whether verification passed. ``None`` when the projection was never verified.

        Not the same question as whether the projection is usable - an unverified projection
        transforms exactly as well as a verified one. It is only a statement about evidence.
        """
        return None if self.report is None else self.report.is_valid

    def scale(self, values):
        """Apply the fitted scaler, if there is one, to raw value channels."""
        values = np.asarray(values, dtype=float)
        if values.shape[1] != self.n_value_channels:
            raise ValueError(
                "projection was fitted on {0} value channel(s), got {1}".format(
                    self.n_value_channels, values.shape[1]))
        if self.scaler is None:
            return values
        return np.asarray(self.scaler.transform(values), dtype=float)

    def transform(self, path):
        """Reduce one ``(n, 1 + d)`` path to ``(n, 1 + n_components)``, time untouched.

        The time channel is passed through rather than embedded: it carries the interval
        slicing and the base-pointing, and a UMAP coordinate would carry neither.

        Note that transforming the fit data does not reproduce the fitted embedding exactly -
        ``UMAP.transform`` re-optimises new points against the existing layout rather than
        looking them up. Everything downstream of here should therefore go through this method,
        including the corpus, so that corpus and test streams are placed by the same instrument.
        """
        path = np.asarray(path, dtype=float)
        embedded = np.asarray(self.mapper.transform(self.scale(path[:, 1:])), dtype=float)
        return np.column_stack([path[:, :1], embedded])

    def transform_frame(self, frame, time_col=None, value_cols=None):
        """Reduce a DataFrame, returning ``Time`` plus ``Component_1..k``."""
        from anomalies_scale.canonical_streams import frame_to_path

        reduced = self.transform(frame_to_path(frame, time_col, value_cols))
        columns = ["Time"] + [COMPONENT_TEMPLATE.format(i + 1)
                              for i in range(self.n_components)]
        return pd.DataFrame(reduced, columns=columns)

    def save(self, path):
        """Pickle the projection, with the report written beside it as readable JSON.

        Pickle because a fitted UMAP holds numba-compiled state and a nearest-neighbour index
        with no portable serialisation of their own. It is version-fragile in consequence - a
        projection saved under one umap-learn will not reliably load under another - so the
        parameters and the report go to JSON as well, where they stay readable when the pickle
        will not load.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)

        sidecar = {"params": self.params,
                   "n_value_channels": self.n_value_channels,
                   "n_components": self.n_components,
                   "reference": self.reference,
                   "stream_names": self.stream_names,
                   "verification": None if self.report is None else self.report.to_dict()}
        path.with_suffix(".json").write_text(
            json.dumps(sidecar, indent=2, default=str), encoding="utf-8")
        return path

    @staticmethod
    def load(path):
        """Read back a projection written by :meth:`save`."""
        with Path(path).open("rb") as handle:
            return pickle.load(handle)

    def __repr__(self):
        return "<UMAPProjection {0} -> {1} channels, {2} slice(s), {3}>".format(
            self.n_value_channels, self.n_components, len(self.mappers),
            {None: "unverified", True: "verified", False: "FAILED verification"}[self.is_valid])


# ---------------------------------------------------------------------------------------
# Entry point


def fit_projection(files=None, streams=None, n_components=DEFAULT_N_COMPONENTS, verify=False,
                   verify_fraction=0.2, thresholds=None, n_neighbors=15, min_dist=0.1,
                   standardise=True, max_fit_rows=4000, relations=None, fit_fraction=None,
                   n_seeds=DEFAULT_N_SEEDS, metric_neighbours=DEFAULT_METRIC_NEIGHBOURS,
                   metric_sample=DEFAULT_METRIC_SAMPLE, drift_probes=4,
                   random_state=0, stream_col=None, time_col=None, value_cols=None,
                   output_path=None, show_progress=False, **umap_kwargs):
    """Fit an aligned UMAP over a corpus of streams, optionally verifying it first.

    Parameters
    ----------
    files : sequence of path, optional
        Corpus files. Normal data only - the projection must never see test streams.
    streams : list of (name, array), optional
        Already-loaded streams, each ``(length, 1 + n_channels)`` with time first. An
        alternative to ``files`` for callers holding the data in memory, and what the
        per-interval detector uses to fit a projection per interval. Exactly one of the two.
    n_components : int
        Value channels to reduce to. The signature dimension that follows is roughly
        ``(1 + n_components) ** trunc``.
    verify : bool
        Withhold a stratified sample of whole streams, fit on the rest, and measure the
        projection against the withheld ones. The projection is returned either way; only
        ``projection.is_valid`` and ``projection.report`` change.
    verify_fraction : float
        Fraction of streams withheld when ``verify`` is set.
    thresholds : dict, optional
        Overrides for :data:`DEFAULT_THRESHOLDS`.
    n_seeds : int
        Independent refits used to measure seed stability during verification. Costs one full
        fit each, and is the dominant expense of a verified run - but it is the only check that
        the embedding is a property of the data rather than of a particular optimiser run. Set
        to 1 to skip it; the report then says so rather than quietly omitting the metrics.
    standardise : bool
        Fit a ``StandardScaler`` on the fit streams and apply it before UMAP. On by default:
        UMAP's neighbour graph is Euclidean and has no Mahalanobis distance to protect it from
        channels on wildly different scales.
    max_fit_rows : int or None
        Cap on rows per slice entering the fit, shared across slices. ``None`` uses every row
        up to the shortest stream's length.
    relations : list of dict, optional
        Correspondences between consecutive slices. Defaults to the identity, which asserts
        streams are synchronised - see the module docstring before relying on it.
    fit_fraction : float, optional
        Fit on only the leading fraction of each stream. Set this to ``split.base_fraction``
        when the dataset ships no separate test files, so the test period still embedded in the
        corpus series does not reach the projection. Projection of the full stream is
        unaffected - this restricts what is *learnt*, not what is transformed.
    output_path : path, optional
        Where to save the fitted projection.

    Returns
    -------
    UMAPProjection

    Notes
    -----
    When ``verify`` is set the returned projection is the one fitted on the *reduced* corpus,
    with the withheld streams excluded. That is deliberate: the projection that ships must be
    the projection that was measured. Refitting on everything afterwards would give a better
    embedding and an untested one, and there is no cheap way to tell how much the extra streams
    changed it.
    """
    from sklearn.preprocessing import StandardScaler

    # `streams` lets a caller that already holds the data in memory skip the file round trip.
    # The per-interval detector needs this: it fits a fresh projection at every interval of the
    # dyadic search, on a slice of an array it is already holding, and writing each slice out
    # to disk only to read it back would dominate the cost of the whole method.
    if streams is None:
        streams = load_corpus_streams(files, stream_col, time_col, value_cols)
    elif files is not None:
        raise ValueError("give either files or streams, not both")

    # When a dataset ships no separate test files, the test period is still sitting at the end
    # of the corpus series and `preprocess` will later cut it off. This stage runs *before*
    # that cut, so without this the projection would be fitted on data it is later asked to
    # judge - a leak that no metric in `verify` could detect, because the withheld streams
    # would be leaked-on too. Truncating each stream to its leading fraction reproduces the
    # split boundary that preprocess will apply.
    if fit_fraction is not None:
        if not 0 < fit_fraction <= 1:
            raise ValueError("fit_fraction must be in (0, 1], got {0}".format(fit_fraction))
        streams = [(name, path[:max(2, int(len(path) * fit_fraction))])
                   for name, path in streams]
        if show_progress and fit_fraction < 1:
            print("fitting on the leading {0:.1%} of each stream, so the test period held "
                  "inside the corpus series does not reach the fit".format(fit_fraction))

    n_value_channels = check_channel_widths(streams)
    if show_progress:
        print("{0} stream(s), {1} value channel(s) -> {2}".format(
            len(streams), n_value_channels, n_components))

    if n_components >= n_value_channels:
        raise ValueError(
            "n_components={0} does not reduce {1} value channel(s); the reduction would add "
            "cost and distortion for nothing".format(n_components, n_value_channels))

    withheld = []
    if verify:
        fit_idx, withheld_idx = stratified_withhold(streams, verify_fraction, random_state)
        withheld = [streams[i] for i in withheld_idx]
        fit_streams = [streams[i] for i in fit_idx]
        if show_progress:
            print("withholding {0} stream(s) for verification, fitting on {1}".format(
                len(withheld), len(fit_streams)))
    else:
        fit_streams = streams

    rows, thinned = shared_row_selection(fit_streams, max_fit_rows, random_state)
    if show_progress and thinned:
        print("fitting on {0} shared row(s) per stream".format(len(rows)))

    raw_slices = [path[rows, 1:] for _, path in fit_streams]

    scaler = None
    if standardise:
        scaler = StandardScaler().fit(np.vstack(raw_slices))
    slices = [np.asarray(scaler.transform(s), dtype=float) if scaler else s
              for s in raw_slices]

    if show_progress:
        print("fitting AlignedUMAP over {0} slice(s)...".format(len(slices)))
    # UMAP's own `verbose` is not tied to show_progress: it prints a per-epoch progress bar for
    # every slice, which on a corpus of any size buries this module's output entirely. Pass
    # verbose=True explicitly if you want it.
    aligned, imputed = fit_aligned(slices, n_components, n_neighbors, min_dist, random_state,
                                   relations=relations, **umap_kwargs)
    check_transform_is_finite(aligned, slices[0])

    total_rows = sum(len(s) for s in slices)
    if imputed and show_progress:
        print("note: {0} of {1} fit row(s) ({2:.2%}) were disconnected and placed at the "
              "centroid".format(imputed, total_rows, imputed / total_rows))

    projection = UMAPProjection(
        aligned, scaler, n_value_channels, n_components, reference=0,
        stream_names=[name for name, _ in fit_streams],
        params={"n_components": n_components, "n_neighbors": n_neighbors,
                "min_dist": min_dist, "standardise": standardise,
                "max_fit_rows": max_fit_rows, "random_state": random_state,
                "verify": bool(verify), "verify_fraction": verify_fraction,
                "disconnected_rows": int(imputed), "fit_rows": int(total_rows)})

    if verify:
        if show_progress:
            print("verifying against {0} withheld stream(s)...".format(len(withheld)))

        stability = None
        if n_seeds and n_seeds > 1:
            probe = withheld[0][1][:, 1:]
            probe = projection.scale(
                probe[subsample_rows(probe, min(500, metric_sample), random_state)])
            if show_progress:
                print("measuring seed stability over {0} fit(s)...".format(n_seeds))
            # Seed 0's embedding is the projection already fitted - reusing it saves one full
            # fit and guarantees the sweep includes the projection actually being returned.
            stability = seed_stability(
                slices, probe, n_components, n_neighbors, min_dist, random_state,
                n_seeds=n_seeds, relations=relations, n_neighbours=metric_neighbours,
                reference_embedding=projection.mapper.transform(probe),
                show_progress=show_progress, **umap_kwargs)

        projection.report = verify_projection(
            projection, withheld, slices, aligned.embeddings_, thresholds=thresholds,
            n_neighbours=metric_neighbours, metric_sample=metric_sample,
            stability=stability, drift_probes=drift_probes, random_state=random_state,
            show_progress=show_progress)
        if imputed:
            projection.report.notes.append(
                "{0} of {1} fit row(s) were disconnected and placed at the centroid".format(
                    imputed, total_rows))
        if show_progress:
            print(projection.report.summary())

    if output_path is not None:
        projection.save(output_path)
        if show_progress:
            print("wrote", output_path)

    return projection


class TimeAlignedProjection:
    """A reduction fitted one embedding per *instant*, aligned along time.

    The alternative slicing to :class:`UMAPProjection`, and a better-founded one when the
    corpus is a set of streams sharing a timebase.

    ``UMAPProjection`` makes each stream a slice, so its rows are instants and the alignment
    relation asserts that row `t` of one stream corresponds to row `t` of the next - i.e. that
    the streams are synchronised. Where that is false the penalty forces unrelated states
    together, which is what ``alignment_drift`` measures and what it caught on SMD's
    overlapping windows at 3.12 against a threshold of 0.25.

    Here each *instant* is a slice instead: slice `t` holds one row per corpus stream, and the
    relation links stream `j` at time `t` to stream `j` at time `t+1`. That correspondence is
    true by construction - it is the same stream one step later - so the alignment penalty is
    doing real work rather than fighting the data. What it buys is an embedding free to follow
    the population's geometry as that geometry drifts over time, tied to its neighbours in time
    rather than pooled across all of them.

    The cost and the risk both come from the same place
    ---------------------------------------------------
    **Cost**: one UMAP fit per instant, so a stream of length `T` needs `T` fits, each over as
    many points as there are corpus streams. That is affordable for windows of a hundred
    points and hundreds of streams; it is not affordable for a 28,479-point stream with 28 of
    them, where each fit would see 28 points.

    **Risk**: consecutive points of a stream are now embedded by *different* mappers. A
    signature integrates increments, so any residual disagreement between mapper `t` and mapper
    `t+1` enters the path directly as length that was never in the data. Under the per-stream
    slicing one mapper handled a whole stream and a smooth path stayed smooth; here the
    alignment penalty is the only thing keeping the path smooth, which makes
    :func:`step_inflation` the metric to watch rather than a formality.

    Time is never embedded. Only the value channels go through UMAP, and the time coordinate is
    reattached to the result, so a ``(n, T, 1 + d)`` block becomes ``(n, T, 1 + k)``.
    """

    def __init__(self, aligned, scaler, n_value_channels, n_components, n_times,
                 params=None, diagnostics=None):
        self.aligned = aligned
        self.scaler = scaler
        self.n_value_channels = int(n_value_channels)
        self.n_components = int(n_components)
        self.n_times = int(n_times)
        self.params = dict(params or {})
        self.diagnostics = dict(diagnostics or {})

    @property
    def mappers(self):
        return self.aligned.mappers_

    def scale(self, values):
        values = np.asarray(values, dtype=float)
        if self.scaler is None:
            return values
        return np.asarray(self.scaler.transform(values), dtype=float)

    def transform(self, paths):
        """Reduce a ``(n, T, 1 + d)`` block of streams to ``(n, T, 1 + k)``.

        Every stream is transformed at once, instant by instant: at time `t` the whole
        population's values go through mapper `t` in a single call. That is both far quicker
        than per-stream transforms and the only correct order, since a mapper belongs to an
        instant rather than to a stream.
        """
        paths = np.asarray(paths, dtype=float)
        if paths.ndim != 3:
            raise ValueError("expected a (n_streams, length, 1 + channels) block, got shape "
                             "{0}".format(paths.shape))
        n, times, width = paths.shape
        if times != self.n_times:
            raise ValueError(
                "projection was fitted on streams of length {0}, got {1}; every stream must "
                "share the timebase the mappers were fitted on".format(self.n_times, times))
        if width - 1 != self.n_value_channels:
            raise ValueError("projection was fitted on {0} value channel(s), got {1}".format(
                self.n_value_channels, width - 1))

        out = np.empty((n, times, 1 + self.n_components), dtype=float)
        out[:, :, 0] = paths[:, :, 0]                       # time, passed through untouched
        for t in range(times):
            embedded = self.mappers[t].transform(self.scale(paths[:, t, 1:]))
            out[:, t, 1:] = np.asarray(embedded, dtype=float)
        return out

    def save(self, path):
        """Pickle the projection, with its parameters beside it as readable JSON.

        Pickle because a fitted UMAP holds numba-compiled state and a pynndescent index, and
        neither has a portable serialisation. That makes the file version-fragile - a
        projection saved under one umap-learn will not reliably load under another - so the
        sidecar records the versions it was written with, and :func:`load_or_fit_time_aligned`
        refuses a mismatch rather than unpickling into undefined behaviour.

        These files are not small: one mapper per instant, each carrying its own copy of the
        corpus rows and neighbour index, so a hundred instants runs to hundreds of megabytes.
        Worth keeping in a cache directory rather than beside the data.
        """
        import umap as _umap

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)

        path.with_suffix(".json").write_text(json.dumps({
            "params": self.params,
            "diagnostics": self.diagnostics,
            "n_value_channels": self.n_value_channels,
            "n_components": self.n_components,
            "n_times": self.n_times,
            "umap_version": getattr(_umap, "__version__", "unknown"),
            "numpy_version": np.__version__,
            "bytes": path.stat().st_size,
        }, indent=2, default=str), encoding="utf-8")
        return path

    @staticmethod
    def load(path):
        """Read back a projection written by :meth:`save`."""
        with Path(path).open("rb") as handle:
            return pickle.load(handle)

    def __repr__(self):
        return "<TimeAlignedProjection {0} -> {1} channels, {2} instants>".format(
            self.n_value_channels, self.n_components, self.n_times)


def fit_time_aligned(streams, n_components=DEFAULT_N_COMPONENTS, n_neighbors=15,
                     min_dist=0.1, standardise=True, random_state=0,
                     alignment_window_size=DEFAULT_ALIGNMENT_WINDOW, show_progress=False,
                     **umap_kwargs):
    """Fit a :class:`TimeAlignedProjection` on a ``(n_streams, T, 1 + d)`` corpus block.

    Corpus streams only - the projection must never see test data, for the same reason the
    whitening must not.

    Parameters
    ----------
    alignment_window_size : int
        How many instants on each side each embedding is aligned against. ``AlignedUMAP``
        allocates ``2 * w + 1`` slots per slice, so ``w = 3`` couples every instant to the
        three before and three after - a seven-slice window centred on itself.

        This is the parameter that governs path smoothness, and directly so. Consecutive
        points of a stream are placed by *adjacent* mappers, so the risk in this slicing is
        that mapper ``t`` and mapper ``t+1`` disagree and their disagreement enters the path
        as length a signature will integrate. The alignment penalty is what prevents that, and
        this is its reach.

        Wider forces more of the trajectory to stay mutually consistent, at the cost of each
        instant being less free to follow its own local geometry, and more alignment work per
        epoch. Narrower does the reverse; at ``w = 1`` each instant is tied only to its
        immediate neighbours, which is the least that still couples the sequence at all.
    """
    umap = import_umap()
    from sklearn.preprocessing import StandardScaler

    streams = np.asarray(streams, dtype=float)
    if streams.ndim != 3:
        raise ValueError("expected (n_streams, length, 1 + channels), got shape {0}".format(
            streams.shape))
    n_streams, times, width = streams.shape
    n_channels = width - 1

    if n_components >= n_channels:
        raise ValueError("n_components={0} does not reduce {1} value channel(s)".format(
            n_components, n_channels))
    if n_streams < 3:
        raise ValueError(
            "each instant is embedded from one point per corpus stream, and {0} stream(s) is "
            "far too few to find any structure in".format(n_streams))

    values = streams[:, :, 1:]
    scaler = StandardScaler().fit(values.reshape(-1, n_channels)) if standardise else None

    slices = []
    for t in range(times):
        block = values[:, t, :]
        slices.append(np.asarray(scaler.transform(block), dtype=float) if scaler else block)

    # Stream `j` at instant `t` is the same stream at instant `t+1`. Unlike the per-stream
    # slicing, this relation is not an assumption about the data - it is what the data is.
    relations = [{j: j for j in range(n_streams)} for _ in range(times - 1)]

    capped = int(min(n_neighbors, max(2, n_streams - 1)))

    # A window reaching past the end of the sequence is not an error - `expand_relations`
    # fills the overhang with -1 and those slots simply do nothing - but it is never what
    # anyone means, and it silently costs alignment work. Capped, and said out loud.
    window = int(alignment_window_size)
    if window < 1:
        raise ValueError(
            "alignment_window_size must be at least 1; {0} would leave the instants "
            "uncoupled, which is a plain UMAP per instant and not an alignment".format(window))
    if window >= times:
        if show_progress:
            print("alignment_window_size {0} exceeds the {1} instants available; "
                  "capping to {2}".format(window, times, max(1, times - 1)))
        window = max(1, times - 1)

    if show_progress:
        print("fitting {0} aligned embedding(s), one per instant, over {1} stream(s) each "
              "(n_neighbors {2}, alignment window +/-{3})".format(
                  times, n_streams, capped, window))

    aligned = umap.AlignedUMAP(
        n_components=n_components, n_neighbors=capped, min_dist=min_dist,
        random_state=random_state, alignment_window_size=window,
        # AlignedUMAP seeds its joint optimisation from transform_seed, not random_state -
        # see fit_aligned. Tying them keeps one knob.
        transform_seed=42 if random_state is None else random_state,
        **umap_kwargs)
    aligned.fit(slices, relations=relations)
    aligned, imputed = adopt_aligned_embeddings(aligned)

    if show_progress and imputed:
        print("  {0} disconnected point(s) placed at their slice centroid".format(imputed))

    return TimeAlignedProjection(
        aligned, scaler, n_channels, n_components, times,
        params={"n_components": n_components, "n_neighbors": capped, "min_dist": min_dist,
                "standardise": standardise, "random_state": random_state,
                "alignment_window_size": window, "n_streams": n_streams, "n_times": times},
        diagnostics={"disconnected_points": int(imputed)})


def projection_fingerprint(streams, **params):
    """A short hash of the data and the parameters that together determine a fit.

    The whole point of a cache here is that reusing the wrong projection is worse than
    refitting: signatures built through one mapping and compared against a corpus built
    through another are not comparable at all, and nothing downstream would notice - the
    distances would simply be wrong. So the key covers the fitted-on array *bit for bit*, not
    a summary of it, alongside every parameter that shapes the result.

    Hashing 14 MB of corpus takes about 30 ms, which is nothing against a 27-minute fit.
    """
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(np.asarray(streams, dtype=np.float64)).tobytes())
    digest.update(json.dumps(sorted((str(k), str(v)) for k, v in params.items())).encode())
    return digest.hexdigest()[:16]


def load_or_fit_time_aligned(streams, cache_dir, show_progress=False, **kwargs):
    """Fit a time-aligned projection, or reuse an identical one already on disk.

    Returns ``(projection, reused)``. The cache key is a hash of the corpus block and the
    fit parameters, so changing either - a different machine, a different split, one more
    component - misses the cache and refits. There is no way to ask for a stale projection
    by accident, which matters more here than the time saved.

    A cached file written under a different umap-learn or numpy is ignored rather than
    loaded: those pickles carry compiled state and version skew produces failures that are
    both obscure and silent.
    """
    import umap as _umap

    cache_dir = Path(cache_dir)
    key = projection_fingerprint(streams, **kwargs)
    path = cache_dir / "time_aligned_{0}.pkl".format(key)
    sidecar = path.with_suffix(".json")

    if path.exists() and sidecar.exists():
        stored = json.loads(sidecar.read_text(encoding="utf-8"))
        same_versions = (
            stored.get("umap_version") == getattr(_umap, "__version__", "unknown")
            and stored.get("numpy_version") == np.__version__)
        if same_versions:
            projection = TimeAlignedProjection.load(path)
            if show_progress:
                print("reusing the projection cached at {0} ({1:.0f} MB)".format(
                    path.name, path.stat().st_size / 1e6))
            return projection, True
        if show_progress:
            print("cache hit at {0} but it was written under umap {1} / numpy {2}; "
                  "refitting".format(path.name, stored.get("umap_version"),
                                     stored.get("numpy_version")))

    projection = fit_time_aligned(streams, show_progress=show_progress, **kwargs)
    projection.save(path)
    if show_progress:
        print("cached the projection at {0} ({1:.0f} MB)".format(
            path.name, path.stat().st_size / 1e6))
    return projection, False


def retained_rank(signatures, keep=0.999):
    """Components carrying `keep` of a signature corpus's variance.

    The check that would have predicted the last failure and did not exist to. A UMAP-reduced
    corpus scored 0.004 point-wise F1 not because its embedding was locally unfaithful - every
    neighbourhood metric passed - but because its *signature* corpus needed 431 components to
    reach 99.9% of variance where the raw one needed 56. The whitening divides by every
    retained direction, so a flat spectrum inflates distances, inflates the calibrated
    threshold, and leaves nothing above it.

    None of the embedding-quality metrics can see this: they judge the embedding, and this is a
    property of the signatures taken *of* the embedding. It costs one SVD.
    """
    signatures = np.asarray(signatures, dtype=float)
    centred = signatures - signatures.mean(axis=0)
    factor = (np.linalg.qr(centred, mode="r")
              if centred.shape[0] > centred.shape[1] else centred)
    values = np.linalg.svd(factor, compute_uv=False)
    if values.size == 0 or values.max() <= 0:
        return 0
    energy = np.cumsum(values ** 2) / np.sum(values ** 2)
    return int(min(np.searchsorted(energy, keep) + 1,
                   (values > 1e-12 * values.max()).sum()))


def path_smoothness(raw_block, latent_block):
    """Mean `step_inflation` over a set of streams, raw against latent.

    The metric that matters most for a time-aligned projection, because consecutive points are
    embedded by different mappers there and any disagreement between them becomes path length
    a signature will integrate. Reported per set rather than per stream: one jagged stream
    among hundreds is noise, a raised mean is the alignment failing.
    """
    ratios = [step_inflation(raw_block[i, :, 1:], latent_block[i, :, 1:])
              for i in range(len(raw_block))]
    ratios = [r for r in ratios if np.isfinite(r)]
    return float(np.mean(ratios)) if ratios else float("nan")


def latent_frame(reduced, n_components):
    """Wrap a projected path as a frame with named columns.

    ``Time`` first, then ``Component_1..k``. The time channel keeps its canonical name so that
    everything downstream - :func:`~anomalies_scale.canonical_streams.resolve_time`, the
    interval slicing, the plotting in `evaluation` - finds it exactly as it would in an
    unprojected stream. A latent stream is still a stream, and nothing after this stage should
    need to know it was reduced.
    """
    columns = ["Time"] + [COMPONENT_TEMPLATE.format(i + 1) for i in range(n_components)]
    return pd.DataFrame(reduced, columns=columns)


def project_streams(files, projection, output_dir, time_col=None, value_cols=None,
                    stream_col=None, show_progress=False):
    """Write every stream in ``files`` into the latent space, one parquet each.

    This is the artifact the rest of the pipeline consumes: the *image* of each stream under
    the projection, as a path over the same time index with ``n_components`` value channels in
    place of the original ones. Signatures are then taken of these, and the whole reason for
    the reduction is that ``siglength`` sees the smaller channel count.

    Written one file per stream rather than one per input file, because a projected stream is
    no longer tied to the file it arrived in and `preprocess` reads a directory of streams.
    Parquet rather than CSV: these are dense float matrices read repeatedly by later stages,
    and the round trip through text costs both precision and time.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    written = []
    for number, file in enumerate(files, start=1):
        for name, path in iter_streams(file, stream_col=stream_col, time_col=time_col,
                                       value_cols=value_cols):
            reduced = projection.transform(path)
            destination = output_dir / "{0}.parquet".format(name)
            latent_frame(reduced, projection.n_components).to_parquet(destination, index=False)
            written.append(destination)

        if show_progress:
            print("  [{0}/{1}] {2}: {3} stream(s) projected".format(
                number, len(files), Path(file).name, len(written)))

    if not written:
        raise ValueError("no streams were projected from {0}".format(
            [str(f) for f in files]))
    return written


def load_stream_block(files, time_col=None, value_cols=None, stream_col=None, label=""):
    """Stack every stream in ``files`` into one ``(n, length, 1 + channels)`` array.

    Equal lengths are required, and not as a convenience: a time-grouped projection holds one
    mapper per instant, so a stream can only be transformed at instants the fit has a mapper
    for. Differing lengths are a data problem to fix upstream, not something this can paper
    over, so they raise here rather than failing obscurely inside ``transform``.
    """
    from anomalies_scale.canonical_streams import is_canonical, iter_paths

    files = [files] if isinstance(files, (str, Path)) else list(files)

    names, paths = [], []
    for file in files:
        # A canonical set already holds one stream per row with time resolved, so it needs
        # neither a stream column nor a time column to be picked out of it.
        source = iter_paths(file) if is_canonical(file) else iter_streams(
            file, stream_col=stream_col, time_col=time_col, value_cols=value_cols)
        for name, path in source:
            names.append(name)
            paths.append(np.asarray(path, dtype=float))

    if not paths:
        raise ValueError("no {0}streams found in {1}".format(
            label + " " if label else "", [str(f) for f in files]))

    lengths = {len(path) for path in paths}
    if len(lengths) > 1:
        raise ValueError(
            "{0}streams have differing lengths {1}. A time-grouped projection fits one "
            "embedding per instant, so every stream must share one timebase.".format(
                label + " " if label else "", sorted(lengths)))

    return names, np.stack(paths)


def check_time_aligned_shape(n_streams, times, n_components, max_instants=2000):
    """Refuse corpus shapes a per-instant fit cannot serve, before spending hours proving it.

    The cost of this scheme is one UMAP fit per instant over one point per corpus stream, so it
    wants **short streams and many of them**. The reverse - long streams, few of them - is
    doubly wrong: thousands of fits, each over too few points to find any structure. Both
    failures are worth catching at the door, because neither announces itself: the first simply
    runs for a day, and the second returns an embedding of noise.
    """
    if times > max_instants:
        raise ValueError(
            "streams are {0} points long, so a per-instant fit would need {0} separate UMAP "
            "fits (the cap is {1}).\n"
            "  This slicing suits short streams and many of them - windows, not whole series. "
            "Window the data first, raise max_instants if you mean it, or set the reduction "
            "off.".format(times, max_instants))

    floor = max(10, 3 * n_components)
    if n_streams < floor:
        raise ValueError(
            "each instant is embedded from one point per corpus stream, and {0} stream(s) is "
            "too few to place in {1} dimensions (want at least {2}).\n"
            "  More corpus streams, fewer components, or no reduction.".format(
                n_streams, n_components, floor))


def project_time_aligned(corpus_files, test_files, projection_path, corpus_dir, test_dir,
                         diagnostics_path=None, n_components=DEFAULT_N_COMPONENTS,
                         n_neighbors=15, min_dist=0.1, standardise=True, random_state=0,
                         alignment_window_size=DEFAULT_ALIGNMENT_WINDOW, cache_dir=None,
                         time_col=None, value_cols=None, stream_col=None,
                         max_instants=2000, show_progress=False):
    """Stage entry point: fit one projection on the corpus, transform both sides, write both.

    This is the whole of the channel reduction as far as the workflows are concerned. Nothing
    downstream is aware of it: `preprocess`, `corpus`, `covariance`, `index`, `calibrate`,
    `score` and the per-interval `detect` all read whatever channels arrive and sign them.
    Because both sides pass through the same fitted object, every signature in the run lives in
    one coordinate system, which is the only property the Mahalanobis distance needs.

    The projection is fitted on **corpus data alone**, for the same reason the whitening is,
    and persisted - it is expensive, it is what was learnt, and a rerun should not refit it.
    """
    corpus_names, corpus_block = load_stream_block(
        corpus_files, time_col, value_cols, stream_col, label="corpus")
    test_names, test_block = load_stream_block(
        test_files, time_col, value_cols, stream_col, label="test")

    n_streams, times, width = corpus_block.shape
    if test_block.shape[1] != times:
        raise ValueError(
            "corpus streams are {0} points and test streams {1}. Both are transformed instant "
            "by instant through the same mappers, so they must share one timebase.".format(
                times, test_block.shape[1]))
    if test_block.shape[2] != width:
        raise ValueError("corpus has {0} channel(s) and test {1}".format(
            width - 1, test_block.shape[2] - 1))

    check_time_aligned_shape(n_streams, times, n_components, max_instants)

    if show_progress:
        print("corpus {0}, test {1}: {2} channel(s) -> {3}".format(
            corpus_block.shape, test_block.shape, width - 1, n_components))

    # Everything that shapes the fit goes in one dict, which is also what the cache key is
    # built from - so a changed alignment window misses the cache and refits rather than
    # silently reusing a projection fitted under a different coupling.
    fit_kwargs = dict(n_components=n_components, n_neighbors=n_neighbors, min_dist=min_dist,
                      standardise=standardise, random_state=random_state,
                      alignment_window_size=alignment_window_size)
    if cache_dir is not None:
        projection, reused = load_or_fit_time_aligned(
            corpus_block, cache_dir, show_progress=show_progress, **fit_kwargs)
    else:
        projection = fit_time_aligned(corpus_block, show_progress=show_progress, **fit_kwargs)
        reused = False

    projection.save(projection_path)

    if show_progress:
        print("transforming both sides...")
    corpus_latent = projection.transform(corpus_block)
    test_latent = projection.transform(test_block)

    written = {
        "corpus": write_latent_streams(corpus_latent, corpus_names, corpus_dir, n_components),
        "test": write_latent_streams(test_latent, test_names, test_dir, n_components),
    }

    diagnostics = {
        "n_corpus_streams": int(n_streams),
        "n_test_streams": int(len(test_names)),
        "n_instants": int(times),
        "n_channels_in": int(width - 1),
        "n_components": int(n_components),
        "projection_reused_from_cache": bool(reused),
        # The two measurements that decide whether this reduction is usable, and the two that
        # earlier attempts failed on. Cheap enough to take every run.
        "path_smoothness_corpus": path_smoothness(corpus_block, corpus_latent),
        "path_smoothness_test": path_smoothness(test_block, test_latent),
    }
    if show_progress:
        print("path smoothness - corpus {0:.3f}, test {1:.3f}  (1.0 = unchanged)".format(
            diagnostics["path_smoothness_corpus"], diagnostics["path_smoothness_test"]))
        print("wrote {0} corpus and {1} test stream(s)".format(
            len(written["corpus"]), len(written["test"])))

    if diagnostics_path is not None:
        diagnostics_path = Path(diagnostics_path)
        diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
        diagnostics_path.write_text(json.dumps(diagnostics, indent=2, default=str),
                                    encoding="utf-8")

    return projection, diagnostics


def write_latent_streams(block, names, output_path, n_components):
    """Write the projected streams as one canonical set.

    Canonical in, canonical out. The reduction is pure preprocessing, and the strongest way to
    say so is for its output to be indistinguishable in shape from its input - a latent stream
    is still a stream, and nothing downstream should be able to tell that a projection happened
    by looking at the file it reads.

    Time is carried through untouched: it is channel 0 of the projected block and becomes the
    canonical `Time` column again, so the latent values stay purely `n_components`-dimensional.
    """
    from anomalies_scale.canonical_streams import to_canonical, write_canonical

    streams = []
    for row, name in zip(block, names):
        streams.append((name, latent_frame(np.asarray(row, dtype=float), n_components)))
    return write_canonical(to_canonical(streams, time_col="Time"), output_path)


#: The two ways the corpus can be sliced for the alignment. Neither is universally right, so
#: both are kept and the choice is configuration rather than a code edit - see
#: :func:`project_corpus_and_test` for what distinguishes them.
SLICINGS = ("per_stream", "per_instant")


def frame_to_block(frame, stream_col=None, time_col=None, value_cols=None):
    """Split a long-format frame of several streams into one ``(n_streams, T, 1 + d)`` block.

    ``stream_col`` names the column that says which stream a row belongs to; without it the
    whole frame is treated as a single stream. Time resolves the same way it does everywhere
    else in the project - :func:`~anomalies_scale.canonical_streams.resolve_time` - so a
    frame that works with the corpus builder works here unchanged.

    Every stream must have the same number of rows. Both slicings need that, for different
    reasons: the time-grouped one has one mapper per instant and cannot place a stream at an
    instant it has no mapper for, and the per-stream one aligns slices that must be the same
    shape. Refused explicitly rather than left to fail inside ``np.stack``.

    Returns
    -------
    (list of str, np.ndarray)
        Stream names in sorted order, and the stacked block with time in channel 0.
    """
    from anomalies_scale.canonical_streams import frame_to_path

    if stream_col is None:
        return ["stream"], np.stack([frame_to_path(frame, time_col, value_cols)])

    if stream_col not in frame.columns:
        raise KeyError("stream column {0!r} is not in the frame; columns are {1}".format(
            stream_col, list(frame.columns)[:8]))

    names, paths = [], []
    for name, group in frame.groupby(stream_col, sort=True):
        columns = [c for c in group.columns if c != stream_col]
        names.append(str(name))
        paths.append(frame_to_path(group[columns], time_col, value_cols))

    lengths = sorted({path.shape[0] for path in paths})
    if len(lengths) > 1:
        raise ValueError(
            "streams have {0} different lengths ({1}); every stream must have the same number "
            "of rows for the alignment to relate them".format(
                len(lengths), ", ".join(str(n) for n in lengths[:5])))

    return names, np.stack(paths)


def project_frame(frame, n_components=DEFAULT_N_COMPONENTS, stream_col=None, time_col=None,
                  value_cols=None, slicing="per_instant", projection=None,
                  show_progress=False, **kwargs):
    """Reduce a frame of streams and return a frame of the same streams, projected.

    The in-memory counterpart to :func:`project_corpus_and_test`, for callers holding their
    data as one long-format DataFrame rather than as a directory of files.

    What the default slicing does
    -----------------------------
    Under ``per_instant`` the frame is split into its individual streams, and the aligned UMAP
    is then fitted with **one slice per time step**: slice `t` holds one row per stream, being
    every stream's state at that instant. The slices are sequenced in time order, and the
    relation between consecutive slices is the identity on streams - stream `j` at time `t` is
    declared to be the same object as stream `j` at time `t + 1`. That correspondence is true
    by construction, which is what makes the alignment penalty do real work here instead of
    forcing unrelated states together.

    ``per_stream`` instead makes each stream a slice and relates them in list order. See
    :func:`project_corpus_and_test`, and check ``alignment_drift`` before relying on it.

    Time is never embedded either way. It is carried through untouched and reattached, so the
    result has ``Time`` plus ``Component_1..k`` and stays sliceable by the same interval logic
    as the raw stream.

    Parameters
    ----------
    projection : optional
        An already-fitted projection to apply. Pass the one returned from the corpus to
        transform test data, so both sides land in the same coordinate system - a signature of
        a stream embedded by any other mapping is not comparable to the corpus signatures.
        Fits a new projection when omitted, which is what corpus data wants.

    Returns
    -------
    (pd.DataFrame, projection)
        The projected frame - one row per stream per instant, carrying ``stream_col`` if one
        was given - and the fitted projection. The projection is returned even when it was
        passed in, so that the object needed to transform test data can never be lost by
        accident on the corpus call.
    """
    if slicing not in SLICINGS:
        raise ValueError("slicing must be one of {0}, got {1!r}".format(SLICINGS, slicing))

    names, block = frame_to_block(frame, stream_col, time_col, value_cols)

    if projection is None:
        if show_progress:
            print("{0} stream(s) of {1} instant(s), {2} channel(s) -> {3} ({4})".format(
                block.shape[0], block.shape[1], block.shape[2] - 1, n_components, slicing))
        if slicing == "per_instant":
            projection = fit_time_aligned(block, n_components=n_components,
                                          show_progress=show_progress, **kwargs)
        else:
            projection = fit_projection(streams=list(zip(names, block)),
                                        n_components=n_components,
                                        show_progress=show_progress, **kwargs)

    if isinstance(projection, TimeAlignedProjection):
        latent = projection.transform(block)
    else:
        latent = np.stack([projection.transform(path) for path in block])

    pieces = []
    for name, rows in zip(names, latent):
        piece = latent_frame(rows, projection.n_components)
        if stream_col is not None:
            piece.insert(0, stream_col, name)
        pieces.append(piece)

    return pd.concat(pieces, ignore_index=True), projection


def project_per_stream(corpus_files, test_files, projection_path, corpus_dir, test_dir,
                       diagnostics_path=None, n_components=DEFAULT_N_COMPONENTS,
                       n_neighbors=15, min_dist=0.1, standardise=True, random_state=0,
                       verify=False, time_col=None, value_cols=None, stream_col=None,
                       show_progress=False, **umap_kwargs):
    """Stage entry point for the per-stream slicing: one aligned mapper per corpus stream.

    The counterpart to :func:`project_time_aligned`, with the same contract - fit on corpus
    data alone, transform both sides through the one fitted object, write both - and the same
    invisibility downstream. Only the slicing differs, and with it what the alignment relation
    is asserting.

    Here slice `i` is corpus stream `i`, its rows are that stream's instants, and the identity
    relation asserts that row `t` of one stream corresponds to row `t` of the next. Whether
    that is true is a property of the corpus, not of the method. It is true for a set of
    independent fixed-length runs sharing a normalised timebase - a fleet of engines, say,
    each window covering the same relative span. It is false for overlapping windows cut from
    one long recording, where row `t` of consecutive windows is a stride apart; that is what
    ``alignment_drift`` measured at 3.12 on SMD against a 0.25 threshold.

    ``alignment_drift`` is therefore reported every run rather than only under ``verify``. It
    is the cheapest available test of the premise, and the premise is the part that fails.

    One practical asymmetry worth knowing: test streams need not match the corpus length here,
    because every stream is placed through a single reference mapper rather than instant by
    instant. The time-aligned slicing has no such freedom.
    """
    corpus_names, corpus_block = load_stream_block(
        corpus_files, time_col, value_cols, stream_col, label="corpus")
    test_names, test_block = load_stream_block(
        test_files, time_col, value_cols, stream_col, label="test")

    n_streams, times, width = corpus_block.shape
    if test_block.shape[2] != width:
        raise ValueError("corpus has {0} channel(s) and test {1}".format(
            width - 1, test_block.shape[2] - 1))

    if show_progress:
        print("corpus {0}, test {1}: {2} channel(s) -> {3}, one slice per stream".format(
            corpus_block.shape, test_block.shape, width - 1, n_components))

    projection = fit_projection(
        streams=list(zip(corpus_names, corpus_block)), n_components=n_components,
        n_neighbors=n_neighbors, min_dist=min_dist, standardise=standardise,
        random_state=random_state, verify=verify, show_progress=show_progress, **umap_kwargs)
    projection.save(projection_path)

    if show_progress:
        print("transforming both sides...")
    corpus_latent = np.stack([projection.transform(path) for path in corpus_block])
    test_latent = np.stack([projection.transform(path) for path in test_block])

    written = {
        "corpus": write_latent_streams(corpus_latent, corpus_names, corpus_dir, n_components),
        "test": write_latent_streams(test_latent, test_names, test_dir, n_components),
    }

    probe_rows = projection.scale(corpus_block[0][:, 1:])
    diagnostics = {
        "slicing": "per_stream",
        "n_corpus_streams": int(n_streams),
        "n_test_streams": int(len(test_names)),
        "n_instants": int(times),
        "n_channels_in": int(width - 1),
        "n_components": int(n_components),
        "path_smoothness_corpus": path_smoothness(corpus_block, corpus_latent),
        "path_smoothness_test": path_smoothness(test_block, test_latent),
        # The premise test. Near zero means the slices agree and the reference mapper the
        # transform uses is representative of all of them; order 1 means it is one arbitrary
        # frame among several incompatible ones and the corpus is not in a single space.
        "alignment_drift": alignment_drift(
            projection.mappers, probe_rows, projection.reference),
        "alignment_drift_threshold": DEFAULT_THRESHOLDS.get("alignment_drift"),
        "verification": None if projection.report is None else projection.report.to_dict(),
    }
    if show_progress:
        print("path smoothness - corpus {0:.3f}, test {1:.3f}  (1.0 = unchanged)".format(
            diagnostics["path_smoothness_corpus"], diagnostics["path_smoothness_test"]))
        print("alignment drift {0:.3f} (threshold {1})".format(
            diagnostics["alignment_drift"], diagnostics["alignment_drift_threshold"]))

    if diagnostics_path is not None:
        diagnostics_path = Path(diagnostics_path)
        diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
        diagnostics_path.write_text(json.dumps(diagnostics, indent=2, default=str),
                                    encoding="utf-8")

    return projection, diagnostics


def project_corpus_and_test(slicing, **kwargs):
    """Dispatch the channel reduction to the configured slicing.

    Both slicings fit on the corpus alone and transform both sides through the one fitted
    object, so everything downstream is identical either way. What differs is which axis the
    alignment runs along, and that is a claim about the corpus rather than a tuning choice:

    ``per_stream``
        One slice per stream, rows are instants. Asserts that streams are synchronised -
        row `t` means the same thing in every stream. One mapper then handles a whole stream,
        so a smooth path stays smooth.

    ``per_instant``
        One slice per instant, rows are streams. Asserts only that a stream at time `t` relates
        to itself at `t+1`, which is true by construction. Consecutive points are placed by
        different mappers, so path smoothness becomes something to measure rather than assume.

    Neither dominates. Pick by asking whether row `t` of two different corpus streams really
    describes the same moment of the same kind of run, and check ``alignment_drift`` and
    ``path_smoothness`` in the diagnostics rather than trusting the answer.
    """
    if slicing not in SLICINGS:
        raise ValueError("slicing must be one of {0}, got {1!r}".format(SLICINGS, slicing))

    if slicing == "per_stream":
        for unsupported in ("alignment_window_size", "cache_dir", "max_instants"):
            kwargs.pop(unsupported, None)
        return project_per_stream(**kwargs)

    kwargs.pop("verify", None)
    return project_time_aligned(**kwargs)


def fit_and_project(corpus_files, projection_path, output_dir, **kwargs):
    """Stage entry point: fit the projection on the corpus, then write its image of it.

    Two outputs, and they serve different halves of the pipeline. The projected corpus feeds
    `preprocess` and everything downstream of it; the saved projection is what `score` needs in
    order to put *test* streams into the same latent space before signing them. A signature of a
    test stream embedded by any other mapping would not be comparable to the corpus signatures,
    and the Mahalanobis distance between them would be meaningless - so the two must come from
    the same fitted object, which is why it is persisted rather than refitted.

    Only corpus files are read here. Test data must not touch the fit, for the same reason it
    must not touch the covariance.
    """
    show_progress = kwargs.get("show_progress", False)
    projection = fit_projection(corpus_files, output_path=projection_path, **kwargs)

    if show_progress:
        print("projecting the corpus into the latent space...")
    written = project_streams(
        corpus_files, projection, output_dir,
        time_col=kwargs.get("time_col"), value_cols=kwargs.get("value_cols"),
        stream_col=kwargs.get("stream_col"), show_progress=show_progress)

    if show_progress:
        print("wrote {0} projected stream(s) to {1}".format(len(written), output_dir))
    return projection, written


def main(argv=None):
    """Fit a projection from the command line."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("files", nargs="+", help="corpus stream files")
    parser.add_argument("--output", required=True, help="where to save the projection")
    parser.add_argument("--n-components", type=int, default=DEFAULT_N_COMPONENTS)
    parser.add_argument("--n-neighbors", type=int, default=15)
    parser.add_argument("--min-dist", type=float, default=0.1)
    parser.add_argument("--verify", action="store_true",
                        help="withhold streams and report embedding quality")
    parser.add_argument("--verify-fraction", type=float, default=0.2)
    parser.add_argument("--n-seeds", type=int, default=DEFAULT_N_SEEDS,
                        help="independent refits used to check seed stability; 1 to skip")
    parser.add_argument("--max-fit-rows", type=int, default=4000)
    parser.add_argument("--no-standardise", action="store_true")
    parser.add_argument("--stream-col", default=None)
    parser.add_argument("--time-col", default=None)
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    projection = fit_projection(
        args.files, n_components=args.n_components, verify=args.verify,
        verify_fraction=args.verify_fraction, n_neighbors=args.n_neighbors,
        n_seeds=args.n_seeds, min_dist=args.min_dist, standardise=not args.no_standardise,
        max_fit_rows=args.max_fit_rows, random_state=args.random_state,
        stream_col=args.stream_col, time_col=args.time_col,
        output_path=args.output, show_progress=not args.quiet)

    if projection.is_valid is False:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
