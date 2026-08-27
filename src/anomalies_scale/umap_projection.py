"""Reduce the channel count with one UMAP embedding and a fitted inverse-free transform.

Why the channel count and not something else
--------------------------------------------
Signature dimension is roughly ``width ** trunc``. It grows polynomially in the length of the
path but *exponentially* in the truncation level, and the base of that exponent is the channel
count - so `d` is the only lever that acts on it. SMD's 39 channels at truncation 4 give
2,374,320 terms, a corpus that cannot be built at all; four latent channels give 780.

What this replaces, and why
---------------------------
The previous stage fitted an ``AlignedUMAP`` - one embedding per slice, coupled by a `relations`
map. That was the wrong tool rather than a mistuned one. AlignedUMAP is built for *overlapping*
slices whose relations identify genuinely shared samples (its own tutorial uses 400-sample
slices advancing by 150); we handed it disjoint slices with relations asserting identity between
different objects, and ``alignment_drift`` read 3.5-4.3 against a 0.25 threshold as a result. It
also has no ``transform`` for unseen data at all, which is what forced the contortions around
test streams.

The replacement is the simplest thing that can work: **one** embedding, fitted once on corpus
data, plus a regressor from ambient coordinates to embedded ones. That regressor is a genuine
function, so corpus and test go through the identical map and the out-of-sample problem
disappears. Nothing is aligned to anything, because there is only one embedding.

Continuity is the requirement, not a nicety
-------------------------------------------
Non-parametric UMAP places points by stochastic optimisation with no continuity constraint, so
two adjacent instants of one stream can land far apart. That matters here more than in most
applications, because the next stage *integrates* the path: jitter in the embedding becomes path
length, and path length is what every level-2 signature term is made of. Fitting a function and
applying it pointwise restores continuity - a kNN regressor is continuous away from its training
points, and its output moves smoothly as its input does.

`path_smoothness` measures exactly this, and is the diagnostic that replaced `alignment_drift`.

Context windows, and how overlapping ones are handled
-----------------------------------------------------
With ``context = m`` each row handed to UMAP is a delay embedding, ``[x_t, x_{t-1}, ...,
x_{t-m+1}]`` - the TimeCluster idea of embedding a *window* rather than an instant, applied at
point level so the output is still one latent point per input point and still a path.

Consecutive rows then share ``m - 1`` observations, and three rules keep that from distorting
the fit:

* **One anchor per observation.** A row is identified by its newest point, so every observation
  anchors exactly one row and receives exactly one latent point. Path length is preserved and
  point indices keep meaning what they meant, which is what lets a flagged interval still be
  read against per-observation labels.
* **Windows never cross a stream boundary.** Rows are built per stream and concatenated
  afterwards. Built over a pooled matrix instead, the rows spanning the join between two engines
  would embed a fictitious transition between unrelated machines.
* **The fit set is strided by `m`.** Overlapping rows are near-duplicates of each other, and a
  neighbour graph built over them would be dominated by trivial temporal adjacency while
  weighting each observation by however many windows happen to contain it. Striding makes the
  fit rows disjoint, so every observation contributes to at most one of them. The *transform*
  still runs on every row - the stride affects what is learnt, never what is emitted.

The first ``m - 1`` points of a stream have no history, and take the stream's own first
observation for the missing slots. That is the convention `preprocessing.delay_columns` already
uses, and it keeps every observation an anchor rather than dropping the opening of every stream.

Where this sits, relative to lead-lag
-------------------------------------
Before it, always. Lead-lag pairs each channel with its own past, taking `k` channels to `2k`;
reducing afterwards would let UMAP mix a channel with its own delayed copy and dissolve the
pairing the transform exists to create. Reducing first gives ``k -> d`` and then ``d -> 2d``,
which is the intended composition and much the cheaper one.

What this stage still cannot tell you
-------------------------------------
Whether anomalies survive the reduction. UMAP is fitted to normal data and an anomaly is off
that manifold, which is exactly where the mapping is least constrained - it may carry anomalies
back onto normal-looking coordinates and erase them. No unsupervised metric settles that; it
needs the detector run on labelled data with and without the reduction, which the pipeline can
now do end to end. Nathan, Nikolaou & Lahav (2025) call this the on-manifold / off-manifold
distinction and argue for running both kinds of detector on one manifold; ``detect.off_manifold``
is this pipeline's linear version of the second arm.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from anomalies_scale.canonical_streams import (
    STREAM_COLUMN, TIME_COLUMN, VALUES_COLUMN, as_matrix, read_canonical, write_canonical)

#: Latent channels when none is given.
DEFAULT_N_COMPONENTS = 4

#: Points of history per embedding input. 1 is the instantaneous case - no delay embedding.
DEFAULT_CONTEXT = 1

#: Neighbours the regressor averages over. Distance-weighted, so this is a smoothing width
#: rather than a hard count: raising it makes latent paths smoother and less faithful.
DEFAULT_KNN_NEIGHBOURS = 10

#: How latent channels are named once they reach the canonical frame.
COMPONENT_TEMPLATE = "Component_{0}"

#: Rows sampled when computing the embedding-quality diagnostics, which are O(n^2).
DIAGNOSTIC_SAMPLE = 2000

#: A latent path whose mean step exceeds this multiple of the ambient one is jittering rather
#: than following the data. Advisory - the stage reports and does not halt.
SMOOTHNESS_THRESHOLD = 3.0


def import_umap():
    """Import umap-learn, with an error that says what to install.

    Deferred because umap-learn pulls in numba and llvmlite, which cost seconds to import and
    are an unnecessary dependency for every run that leaves the reduction switched off.
    """
    try:
        import umap
    except ImportError as error:  # pragma: no cover - environment dependent
        raise ImportError(
            "the channel reduction needs umap-learn, which is not installed. Either "
            "`pip install umap-learn`, or set umap.dimension to -1 to switch the stage off."
        ) from error
    return umap


# ---------------------------------------------------------------------------------------
# Building the matrix UMAP is fitted on
# ---------------------------------------------------------------------------------------
def context_rows(values, context):
    """Delay-embed one stream: ``(length, k)`` becomes ``(length, context * k)``.

    Row *t* is ``[x_t, x_{t-1}, ..., x_{t-context+1}]``, so the row is anchored at its newest
    point and there is exactly one row per observation. Slots are ordered newest-first and
    consistently, so a given observation occupies slot *j* of the row *j* steps after it - the
    correspondence between rows and observations is the same everywhere in the stream.

    The opening ``context - 1`` rows have no history and take ``values[0]`` for the missing
    slots, matching `preprocessing.delay_columns`. Dropping them instead would shorten every
    stream and put point indices out of step with any per-observation labels.
    """
    values = np.asarray(values, dtype=float)
    context = int(context)
    if context < 1:
        raise ValueError("umap.context must be at least 1, got {0}".format(context))
    if context == 1:
        return values
    if not len(values):
        raise ValueError("cannot build context rows for an empty stream")

    slots = [values]
    for lag in range(1, context):
        shifted = np.empty_like(values)
        shifted[:lag] = values[0]
        shifted[lag:] = values[:-lag]
        slots.append(shifted)
    return np.hstack(slots)


def load_streams(source):
    """Read a canonical set as ``[(name, time, values), ...]``."""
    frame = read_canonical(source)
    streams = []
    for row in frame.itertuples(index=False):
        streams.append((str(getattr(row, STREAM_COLUMN)),
                        np.asarray(getattr(row, TIME_COLUMN), dtype=float),
                        as_matrix(getattr(row, VALUES_COLUMN))))
    if not streams:
        raise ValueError("{0} holds no streams".format(source))

    widths = {values.shape[1] for _, _, values in streams}
    if len(widths) > 1:
        raise ValueError(
            "streams disagree about channel count ({0}); one projection cannot serve "
            "them".format(sorted(widths)))
    return streams


def fit_matrix(streams, context, stride=None):
    """The rows one embedding is fitted on, disjoint by construction.

    Built per stream and concatenated, never over a pooled matrix - rows spanning the join
    between two engines would embed a transition that never happened.

    `stride` defaults to `context`, which makes the fit rows share no observation with one
    another. Overlapping rows are near-duplicates, and a neighbour graph over them is dominated
    by trivial temporal adjacency while weighting each observation by however many windows
    contain it. Only the fit is strided; every row is still transformed.
    """
    stride = int(stride) if stride else int(context)
    if stride < 1:
        raise ValueError("umap.fit_stride must be at least 1, got {0}".format(stride))

    blocks = [context_rows(values, context)[::stride] for _, _, values in streams]
    return np.vstack(blocks)


# ---------------------------------------------------------------------------------------
# The fitted projection
# ---------------------------------------------------------------------------------------
class Projection:
    """A scaler and a regressor: everything needed to place a point in the latent space.

    The UMAP object itself is *not* kept. Once the regressor has been fitted to its embedding,
    the embedding is what matters and umap-learn is not needed to apply it - so a projection
    loads without numba and transforms in microseconds.

    Corpus streams go through this same regressor, not through UMAP's own embedding of them.
    Using the raw embedding for the corpus and the regressor for test data would place the two
    sides in subtly different coordinate systems, and a Mahalanobis distance between signatures
    of paths in different coordinates means nothing.
    """

    def __init__(self, scaler, regressor, context, n_components, params=None):
        self.scaler = scaler
        self.regressor = regressor
        self.context = int(context)
        self.n_components = int(n_components)
        self.params = params or {}

    def transform(self, values):
        """Place one stream's ``(length, k)`` values into ``(length, n_components)``."""
        rows = context_rows(values, self.context)
        if self.scaler is not None:
            rows = self.scaler.transform(rows)
        return np.asarray(self.regressor.predict(rows), dtype=float)

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            pickle.dump({"scaler": self.scaler, "regressor": self.regressor,
                         "context": self.context, "n_components": self.n_components,
                         "params": self.params}, handle)
        return path

    @classmethod
    def load(cls, path):
        with Path(path).open("rb") as handle:
            stored = pickle.load(handle)
        return cls(stored["scaler"], stored["regressor"], stored["context"],
                   stored["n_components"], stored.get("params"))


def fit_embedding(ambient, n_components, n_neighbors=15, min_dist=0.1, standardise=True,
                  random_state=0, knn_neighbours=DEFAULT_KNN_NEIGHBOURS, context=1):
    """Embed a matrix once, then fit the regressor that turns the embedding into a function.

    Shared by both reductions - of channels, and of signature terms - because the awkward part
    is identical either way: UMAP gives coordinates for the rows it saw and no way to place a
    row it did not, and everything in this pipeline needs corpus and test to land in the same
    coordinates.

    Returns
    -------
    (Projection, dict)
        The projection, and the embedding it was fitted to plus diagnostics.
    """
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.preprocessing import StandardScaler

    umap = import_umap()
    ambient = np.asarray(ambient, dtype=float)

    scaler = None
    if standardise:
        # UMAP builds its neighbour graph under a plain Euclidean metric, so one coordinate
        # measured in thousands would decide the embedding by itself. That is not a corner case
        # for signature vectors: a level-2 term scales like the square of the increment, and the
        # variance ratio between levels 1 and 2 measured 1,363x on C-MAPSS.
        scaler = StandardScaler().fit(ambient)
        ambient = scaler.transform(ambient)

    # Capped: UMAP cannot ask for more neighbours than there are other points.
    neighbours = int(min(n_neighbors, max(2, ambient.shape[0] - 1)))
    reducer = umap.UMAP(n_components=int(n_components), n_neighbors=neighbours,
                        min_dist=float(min_dist), random_state=random_state)
    embedded = np.asarray(reducer.fit_transform(ambient), dtype=float)

    # Distance-weighted so the map is continuous away from the fit rows. Uniform weights would
    # make it piecewise constant, and a staircase latent path is exactly the jitter the
    # signature would integrate.
    regressor = KNeighborsRegressor(
        n_neighbors=int(min(knn_neighbours, ambient.shape[0])), weights="distance")
    regressor.fit(ambient, embedded)

    projection = Projection(
        scaler, regressor, context, n_components,
        params={"n_neighbors": neighbours, "min_dist": float(min_dist),
                "standardise": bool(standardise), "random_state": random_state,
                "knn_neighbours": int(min(knn_neighbours, ambient.shape[0]))})

    # How faithfully the regressor reproduces the embedding it was fitted to. This is the price
    # of fitting a function rather than embedding directly, and the number that says whether it
    # was worth paying: the regressor, not UMAP, is what everything actually goes through.
    predicted = np.asarray(regressor.predict(ambient), dtype=float)
    spread = float(np.linalg.norm(embedded - embedded.mean(axis=0), axis=1).mean())
    fidelity = float(np.linalg.norm(predicted - embedded, axis=1).mean() / max(spread, 1e-12))

    return projection, {"n_fit_rows": int(ambient.shape[0]),
                        "ambient_columns": int(ambient.shape[1]),
                        "knn_relative_error": fidelity,
                        "embedding": embedded, "fit_ambient": ambient}


def fit_projection(streams, n_components=DEFAULT_N_COMPONENTS, context=DEFAULT_CONTEXT,
                   n_neighbors=15, min_dist=0.1, standardise=True, random_state=0,
                   fit_stride=None, knn_neighbours=DEFAULT_KNN_NEIGHBOURS,
                   show_progress=False):
    """Fit the channel reduction: one UMAP over context rows, plus its regressor."""
    ambient = fit_matrix(streams, context, fit_stride)
    if show_progress:
        print("fitting on {0:,} row(s) x {1} column(s) from {2} stream(s) "
              "(context {3}, stride {4})".format(
                  ambient.shape[0], ambient.shape[1], len(streams), context,
                  fit_stride or context))

    projection, info = fit_embedding(
        ambient, n_components, n_neighbors, min_dist, standardise, random_state,
        knn_neighbours, context)
    projection.params["fit_stride"] = int(fit_stride or context)
    return projection, info


# ---------------------------------------------------------------------------------------
# Reducing the signature space instead of the channel space
# ---------------------------------------------------------------------------------------
#: How a latent signature component is named, so a reduced corpus is still a corpus.
LATENT_TEMPLATE = "sig_{0}"


def signature_columns(frame):
    """Split a corpus frame into its identity columns and its signature columns."""
    signature = [c for c in frame.columns if str(c).startswith("sig_")]
    identity = [c for c in frame.columns if c not in set(signature)]
    if not signature:
        raise ValueError(
            "this corpus has no 'sig_*' columns to reduce; its columns are {0}".format(
                list(frame.columns)[:8]))
    return identity, signature


def fit_signature_projection(corpus, n_components, n_neighbors=15, min_dist=0.1,
                             random_state=0, knn_neighbours=DEFAULT_KNN_NEIGHBOURS,
                             fit_rows=None, show_progress=False):
    """Embed the signature corpus itself, and return the function that places new signatures.

    The other order. `fit_projection` reduces channels *before* signing, which is the lever that
    acts on ``width ** trunc`` and the only one that makes a high truncation affordable at all.
    This reduces the signature vectors *after* signing, which is linear rather than exponential -
    the corpus still has to be built at full width - but it leaves the path untouched, so
    whatever the signature captured is captured before anything is discarded.

    That difference matters for what survives. Reducing channels first throws information away
    while it is still a path, and the signature can only integrate what is left; measured on
    C-MAPSS it cost precision heavily, 0.853 down to about 0.50. Reducing afterwards lets the
    signature see every channel and only then compresses the description of it.

    One embedding over every depth, not one per depth. Wide intervals produce systematically
    larger vectors than narrow ones, so depth is a real axis of variation in this matrix and the
    embedding will encode some of it - but the search is banded by depth and the thresholds are
    per depth, so that information is redundant rather than harmful. Fitting per depth would
    avoid spending latent dimensions on it, at the cost of a projection per band.
    """
    from anomalies_scale.covariance_creation import read_corpus

    frame = read_corpus(corpus)
    _, signature = signature_columns(frame)
    ambient = frame[signature].to_numpy(dtype=float)

    rows = np.arange(len(ambient))
    if fit_rows and len(rows) > int(fit_rows):
        # Sampled rather than truncated: the corpus is ordered by depth, so a head slice would
        # fit the embedding on the widest intervals alone.
        rows = np.random.default_rng(random_state).choice(rows, int(fit_rows), replace=False)

    if show_progress:
        print("reducing the signature space: {0:,} term(s) -> {1}, fitted on {2:,} of {3:,} "
              "interval(s)".format(len(signature), n_components, len(rows), len(ambient)))

    projection, info = fit_embedding(
        ambient[rows], n_components, n_neighbors, min_dist, standardise=True,
        random_state=random_state, knn_neighbours=knn_neighbours, context=1)
    info["n_terms"] = len(signature)
    info["n_intervals"] = int(len(ambient))
    return projection, info


def reduce_corpus(corpus, projection, output_path=None, show_progress=False):
    """Rewrite a signature corpus in latent coordinates, keeping its identity columns.

    The result is still a corpus - `depth`, `stream`, `lo`, `hi`, `n_cells`, then ``sig_*`` -
    so `covariance`, `index` and `calibrate` read it without knowing anything changed. Only
    `score` needs the projection, to place the signatures it computes on the fly.
    """
    from anomalies_scale.covariance_creation import read_corpus

    frame = read_corpus(corpus)
    identity, signature = signature_columns(frame)
    latent = projection.transform(frame[signature].to_numpy(dtype=float))

    reduced = pd.DataFrame(
        latent, columns=[LATENT_TEMPLATE.format(i) for i in range(latent.shape[1])],
        index=frame.index)
    reduced = pd.concat([frame[identity], reduced], axis=1)

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.suffix.lower() == ".csv":
            reduced.to_csv(output_path, index=False)
        else:
            reduced.to_parquet(output_path, index=False)
        if show_progress:
            print("wrote {0:,} interval(s) x {1} latent term(s) to {2}".format(
                len(reduced), latent.shape[1], output_path))
    return reduced


def reduce_signature_corpus(corpus, output_corpus, projection_path, n_components,
                            diagnostics_path=None, n_neighbors=15, min_dist=0.1,
                            random_state=0, knn_neighbours=DEFAULT_KNN_NEIGHBOURS,
                            fit_rows=None, show_progress=False):
    """Stage entry point: fit on the corpus, rewrite it, and persist the projection."""
    projection, info = fit_signature_projection(
        corpus, n_components, n_neighbors, min_dist, random_state, knn_neighbours,
        fit_rows, show_progress)

    reduce_corpus(corpus, projection, output_corpus, show_progress)
    projection.save(projection_path)

    diagnostics = {
        "n_components": int(n_components),
        "n_terms": info["n_terms"],
        "n_intervals": info["n_intervals"],
        "n_fit_rows": info["n_fit_rows"],
        "knn_relative_error": info["knn_relative_error"],
        "trustworthiness": embedding_quality(
            info["fit_ambient"], info["embedding"], n_neighbors, random_state=random_state),
        "params": projection.params,
    }
    if diagnostics_path is not None:
        diagnostics_path = Path(diagnostics_path)
        diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
        diagnostics_path.write_text(json.dumps(diagnostics, indent=2, default=str),
                                    encoding="utf-8")
    if show_progress:
        print("trustworthiness {trustworthiness:.3f}; kNN relative error "
              "{knn_relative_error:.3%}".format(**diagnostics))
    return diagnostics


# ---------------------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------------------
def path_smoothness(ambient, latent):
    """Mean latent step length over mean ambient step length, both scale-normalised.

    The diagnostic that replaced `alignment_drift`, and it measures the thing that actually
    matters downstream: the signature integrates the path, so an embedding that jitters between
    adjacent instants inflates every level-2 term with length the data does not have. A value
    near 1 says the latent path moves as the ambient one does; a large value says it does not.
    """
    ambient, latent = np.asarray(ambient, dtype=float), np.asarray(latent, dtype=float)
    if len(ambient) < 2:
        return float("nan")

    def normalised_step(block):
        spread = np.linalg.norm(block.std(axis=0)) or 1.0
        return float(np.linalg.norm(np.diff(block, axis=0), axis=1).mean() / spread)

    ambient_step = normalised_step(ambient)
    return float(normalised_step(latent) / ambient_step) if ambient_step else float("nan")


def embedding_quality(ambient, embedded, n_neighbours=15, sample=DIAGNOSTIC_SAMPLE,
                      random_state=0):
    """Trustworthiness of the embedding, on a subsample because the metric is O(n^2)."""
    from sklearn.manifold import trustworthiness

    rows = np.arange(len(ambient))
    if len(rows) > sample:
        rows = np.random.default_rng(random_state).choice(rows, sample, replace=False)
    neighbours = int(min(n_neighbours, max(2, len(rows) // 3)))
    return float(trustworthiness(ambient[rows], embedded[rows], n_neighbors=neighbours))


# ---------------------------------------------------------------------------------------
# Stage entry point
# ---------------------------------------------------------------------------------------
def latent_frame(streams, projection, show_progress=False, label=""):
    """Project every stream and return a canonical frame of latent streams."""
    rows, ratios = [], []
    for name, time, values in streams:
        latent = projection.transform(values)
        ratios.append(path_smoothness(values, latent))
        rows.append({STREAM_COLUMN: name, TIME_COLUMN: time, VALUES_COLUMN: latent})

    if show_progress:
        finite = [r for r in ratios if np.isfinite(r)]
        print("  {0}: {1} stream(s) projected, smoothness ratio median {2:.2f}, max "
              "{3:.2f}".format(label or "projected", len(rows),
                               float(np.median(finite)) if finite else float("nan"),
                               float(np.max(finite)) if finite else float("nan")))
    return pd.DataFrame(rows), ratios


def project_corpus_and_test(corpus_files, test_files, projection_path, corpus_dir, test_dir,
                            diagnostics_path=None, n_components=DEFAULT_N_COMPONENTS,
                            context=DEFAULT_CONTEXT, n_neighbors=15, min_dist=0.1,
                            standardise=True, random_state=0, fit_stride=None,
                            knn_neighbours=DEFAULT_KNN_NEIGHBOURS, show_progress=False):
    """Fit on the corpus, project both sides, and write everything the run needs.

    Both sides are projected here rather than only the corpus, because the reduction is pure
    preprocessing: every stage after it reads latent streams and never learns that a reduction
    happened. The projection is still persisted - it is what was learnt, and a rerun on
    unchanged inputs should not refit it.
    """
    corpus = load_streams(corpus_files)
    if show_progress:
        print("channel reduction: {0} -> {1} latent channel(s)".format(
            corpus[0][2].shape[1], n_components))

    projection, info = fit_projection(
        corpus, n_components=n_components, context=context, n_neighbors=n_neighbors,
        min_dist=min_dist, standardise=standardise, random_state=random_state,
        fit_stride=fit_stride, knn_neighbours=knn_neighbours, show_progress=show_progress)

    corpus_frame, corpus_ratios = latent_frame(corpus, projection, show_progress, "corpus")
    write_canonical(corpus_frame, corpus_dir)

    test_ratios = []
    if test_files:
        test = load_streams(test_files)
        test_frame, test_ratios = latent_frame(test, projection, show_progress, "test")
        write_canonical(test_frame, test_dir)

    projection.save(projection_path)

    ratios = [r for r in corpus_ratios + test_ratios if np.isfinite(r)]
    diagnostics = {
        "n_components": int(n_components),
        "context": int(context),
        "ambient_channels": int(corpus[0][2].shape[1]),
        "n_fit_rows": info["n_fit_rows"],
        "ambient_columns": info["ambient_columns"],
        "knn_relative_error": info["knn_relative_error"],
        "trustworthiness": embedding_quality(
            info["fit_ambient"], info["embedding"], n_neighbors if n_neighbors else 15,
            random_state=random_state),
        "smoothness_median": float(np.median(ratios)) if ratios else None,
        "smoothness_max": float(np.max(ratios)) if ratios else None,
        "smoothness_threshold": SMOOTHNESS_THRESHOLD,
        "params": projection.params,
    }
    # Advisory, never fatal. Whether a poor embedding is acceptable is a judgement about the
    # data, and the person reading the report is better placed to make it than the pipeline.
    diagnostics["verdict"] = (
        "ok" if ratios and np.median(ratios) <= SMOOTHNESS_THRESHOLD else "check")

    if diagnostics_path is not None:
        diagnostics_path = Path(diagnostics_path)
        diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
        diagnostics_path.write_text(json.dumps(diagnostics, indent=2, default=str),
                                    encoding="utf-8")

    if show_progress:
        print("trustworthiness {trustworthiness:.3f}; kNN relative error "
              "{knn_relative_error:.3%}; smoothness median {smoothness_median:.2f} "
              "({verdict})".format(**diagnostics))
        print("wrote {0}".format(projection_path))

    return diagnostics
