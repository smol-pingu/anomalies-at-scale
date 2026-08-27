"""Feature bagging: many detectors on random channel subsets, combined by vote.

The random subspace method, applied to the whole detector rather than to a classifier. Each draw
picks ``round(sqrt(d))`` of the *d* value channels, builds a corpus from those channels alone,
fits a metric to it, calibrates it and searches the test streams with it. A point's vote count is
how many draws flagged it, and the verdict is ``votes >= detect.bagged.votes``, where 1 is
exactly the union.

Why it is not just another `detect.method`
------------------------------------------
The other detectors differ in the *question* they ask of a corpus interval - distance to the
nearest normal, or how easily isolated - and all read one corpus. Bagging changes the corpus
itself: a draw signs a different set of channels, so its signature terms are different
quantities and its whitening lives in a different space. Every draw therefore needs its own
corpus, covariance, index and thresholds, and none of them are comparable with another draw's.
Only the rasterised verdicts are, which is why the aggregation happens on masks rather than on
scores.

Why it should help
------------------
A metric fitted on all *d* channels dilutes an anomaly that lives in a few of them. Measured on
SMD machine-1-1, where the interpretation labels say each anomaly implicates 7 to 31 of the 38
channels: 30 draws of 6 channels gave a union F1 of 0.521 against 0.398 and 0.423 for the
single full-width detectors, with individual draws ranging from 0.015 to 0.583. The spread is
the mechanism - the ensemble is aggregating draws that happened to contain the affected channels
with draws that missed them entirely.

It is also cheaper per draw than it looks. Signature dimension is roughly ``width ** trunc``, so
C-MAPSS's 24 value channels plus time give 650 terms at truncation 2, while a draw of
``round(sqrt(24)) = 5`` channels plus time gives 6 + 36 = 42. A draw is a far smaller problem
than the full-width detector, and better conditioned: on SMD the retained rank per draw ran 4 to
15, against a full-width fit that needed heavy truncation.

The vote count is also a score
------------------------------
``votes / draws`` is a continuous quantity in [0, 1] for every point, which is exactly what
ROC-AUC and PR-AUC need - so the bagged detector supplies its own ranking without the separate
sliding-window pass that :mod:`anomalies_scale.AUC` performs for the single-detector path.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from anomalies_scale.canonical_streams import (
    STREAM_COLUMN, TIME_COLUMN, VALUES_COLUMN, as_matrix, read_canonical)
from anomalies_scale.covariance_creation import covariance_matrix, signature_matrix
from anomalies_scale.crossvalidated_thresholding import (
    parse_statistic, source_of, stream_folds)
from anomalies_scale.signature_computer import compute_corpus
from anomalies_scale.stream_scoring import ANOMALOUS_COLUMN, score_streams

#: Value of `detect.bagged.draws` that switches the stage off, matching `window.size`.
BAGGING_OFF = -1


def subset_size(n_channels, features=None):
    """Channels per draw. ``None`` gives the random subspace default of round(sqrt(d))."""
    if features is not None:
        return max(1, min(int(features), int(n_channels)))
    return max(1, min(int(round(np.sqrt(n_channels))), int(n_channels)))


def channel_draws(n_channels, draws, features=None, random_state=0):
    """Which channels each draw sees, as sorted index arrays.

    Sampled without replacement *within* a draw and independently *between* them, so a channel
    may appear in many draws or none. That is the point: coverage is what the vote aggregates
    over, and forcing every channel to appear equally often would make the draws dependent.
    """
    size = subset_size(n_channels, features)
    rng = np.random.default_rng(random_state)
    return [np.sort(rng.choice(int(n_channels), size=size, replace=False))
            for _ in range(int(draws))]


def restrict_channels(frame, columns):
    """A canonical set carrying only the given value channels.

    `Time` is untouched - it is not one of the *d* channels being sampled, and dropping it would
    change what a signature means rather than which channels it sees.
    """
    out = frame.copy()
    out[VALUES_COLUMN] = [np.ascontiguousarray(as_matrix(block)[:, columns])
                          for block in frame[VALUES_COLUMN]]
    return out


def intervals_to_mask(intervals, length):
    mask = np.zeros(int(length), dtype=bool)
    for pair in intervals if intervals is not None else []:
        lo, hi = int(pair[0]), int(pair[1])
        mask[max(lo, 0):min(hi, length - 1) + 1] = True
    return mask


def mask_to_intervals(mask):
    """Contiguous True runs as inclusive ``[lo, hi]`` pairs - the scorer's own convention."""
    edges = np.diff(np.concatenate([[0], np.asarray(mask).astype(int), [0]]))
    return [[int(lo), int(hi) - 1]
            for lo, hi in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))]


def one_draw(corpus_streams, test_streams, columns, trunc, granularity, variance_keep,
             band, folds, statistic, random_state, span, sig_tol, tol, neighbours,
             detector, forest_settings):
    """Build, calibrate and run one detector on one channel subset.

    Everything is in memory. A draw's corpus, metric and index are meaningless to any other
    draw - different channels, different terms, a different whitened space - so persisting them
    would cost disk for artifacts nothing can reuse.
    """
    import faiss

    from anomalies_scale.index_creation import PooledIndex

    corpus = compute_corpus(restrict_channels(corpus_streams, columns),
                            trunc=trunc, granularity=granularity)
    signatures, terms = signature_matrix(corpus)
    depths = corpus["depth"].to_numpy(dtype=int)
    streams = np.asarray([source_of(name) for name in corpus[STREAM_COLUMN].to_numpy()])

    matrix, info = covariance_matrix(signatures, variance_keep=variance_keep, form="inv_sqrt")
    # The factor rides along on the diagnostics. A draw is already a small problem - 42 terms at
    # 5 channels - so the latent projection saves little here, but using it keeps every detector
    # in the pipeline reaching its space the same way.
    metric = info["whitening"]
    whitened = np.ascontiguousarray(metric.apply(signatures), dtype="float32")

    if detector == "isolation_forest":
        from anomalies_scale.isolation_detector import (
            DepthForests, crossvalidated_forest_thresholds, fit_forests)

        thresholds, _ = crossvalidated_forest_thresholds(
            corpus, matrix, k=folds, statistic=statistic, band=band,
            space=forest_settings["space"], n_estimators=forest_settings["n_estimators"],
            max_samples=forest_settings["max_samples"], random_state=random_state,
            windowed=True)
        engine = fit_forests(corpus, matrix, band=band, space=forest_settings["space"],
                             n_estimators=forest_settings["n_estimators"],
                             max_samples=forest_settings["max_samples"],
                             random_state=random_state)
        metric = engine.scorer_covariance
    else:
        from anomalies_scale.crossvalidated_thresholding import fold_distances

        distances = fold_distances(whitened, depths, streams,
                                   stream_folds(streams, folds, random_state), band, neighbours)
        reduce_to_threshold = parse_statistic(statistic)
        thresholds = {int(d): float(reduce_to_threshold(distances[depths == d]))
                      for d in np.unique(depths)}

        index = faiss.IndexFlatL2(whitened.shape[1])
        index.add(whitened)
        engine = PooledIndex(index, depth=depths, split=np.zeros(len(depths), dtype=int),
                             band=band, terms=terms)

    scored = score_streams(restrict_channels(test_streams, columns), engine, metric,
                           threshold=thresholds, span=span, sig_tol=sig_tol, tol=tol,
                           neighbours=neighbours, show_progress=False)
    return scored, {"terms": int(signatures.shape[1]), "rank": int(info["rank"]),
                    "intervals": int(len(corpus))}


def run_bagged(corpus_streams, test_streams, draws=30, features=None, votes=1,
               trunc=2, granularity=3, variance_keep=0.999, band=1, folds=5, statistic="p95",
               random_state=0, span=None, sig_tol=2, tol=0, neighbours=1,
               detector="mahalanobis", forest_settings=None, output_path=None,
               scores_path=None, diagnostics_path=None, show_progress=False):
    """Every draw, aggregated into one verdict per point.

    Returns
    -------
    (pd.DataFrame, pd.DataFrame, dict)
        The scored frame in the shape `score_streams` produces, the per-point vote fractions,
        and per-draw diagnostics.
    """
    corpus_streams = read_canonical(corpus_streams)
    test_streams = read_canonical(test_streams)
    forest_settings = forest_settings or {"space": "whitened", "n_estimators": 200,
                                          "max_samples": "auto"}

    n_channels = as_matrix(corpus_streams[VALUES_COLUMN].iloc[0]).shape[1]
    subsets = channel_draws(n_channels, draws, features, random_state)
    lengths = {str(getattr(row, STREAM_COLUMN)): len(as_matrix(getattr(row, VALUES_COLUMN)))
               for row in test_streams.itertuples(index=False)}

    if show_progress:
        print("{0} draw(s) of {1} of {2} value channel(s); {3} detector, votes >= {4}".format(
            draws, len(subsets[0]), n_channels, detector, votes))

    tally = {name: np.zeros(length, dtype=int) for name, length in lengths.items()}
    per_draw = []

    for number, columns in enumerate(subsets, start=1):
        started = time.time()
        scored, info = one_draw(corpus_streams, test_streams, columns, trunc, granularity,
                                variance_keep, band, folds, statistic,
                                random_state + number, span, sig_tol, tol, neighbours,
                                detector, forest_settings)

        flagged = 0
        for row in scored.itertuples(index=False):
            name = str(getattr(row, STREAM_COLUMN))
            mask = intervals_to_mask(getattr(row, ANOMALOUS_COLUMN), lengths[name])
            tally[name] += mask
            flagged += int(mask.sum())

        total = sum(lengths.values())
        info.update(draw=number, channels=" ".join(str(c) for c in columns),
                    flagged_fraction=flagged / total if total else 0.0,
                    seconds=time.time() - started)
        per_draw.append(info)
        if show_progress:
            print("  draw {draw:>3}  {terms:>4} terms  rank {rank:>3}  flagged "
                  "{flagged_fraction:>6.2%}  ({seconds:.0f}s)".format(**info), flush=True)

    # --- aggregate ---------------------------------------------------------------------------
    out = test_streams.copy()
    verdicts, fractions, counts, points = [], [], [], []
    for row in test_streams.itertuples(index=False):
        name = str(getattr(row, STREAM_COLUMN))
        count = tally[name]
        mask = count >= int(votes)
        verdicts.append(mask_to_intervals(mask))
        # votes / draws is a continuous score per point, which is what a ranking metric needs.
        fractions.append(count / float(draws))
        counts.append(int(mask.sum()))
        points.append(len(count))

    out[ANOMALOUS_COLUMN] = verdicts
    out["n_anomalous_intervals"] = [len(v) for v in verdicts]
    out["n_anomalous_points"] = counts
    out["anomalous_fraction"] = [c / p if p else 0.0 for c, p in zip(counts, points)]
    out["max_score"] = [float(f.max()) if len(f) else 0.0 for f in fractions]

    votes_frame = pd.DataFrame({STREAM_COLUMN: list(out[STREAM_COLUMN]),
                                "point_score": fractions})

    diagnostics = {
        "draws": int(draws), "features": int(len(subsets[0])), "n_channels": int(n_channels),
        "votes": int(votes), "detector": detector,
        "flagged_fraction": float(sum(counts) / max(sum(points), 1)),
        "per_draw": per_draw,
        "mean_draw_flagged": float(np.mean([d["flagged_fraction"] for d in per_draw])),
        "mean_rank": float(np.mean([d["rank"] for d in per_draw])),
        "seconds": float(sum(d["seconds"] for d in per_draw)),
    }

    if show_progress:
        print("aggregated: {0:.2%} of points flagged at votes >= {1}; a single draw flagged "
              "{2:.2%} on average".format(diagnostics["flagged_fraction"], votes,
                                          diagnostics["mean_draw_flagged"]))

    for path, payload in ((scores_path, out), (output_path, votes_frame)):
        if path is None:
            continue
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        storable = payload.copy()
        for column in (TIME_COLUMN, VALUES_COLUMN):
            if column in storable.columns:
                storable[column] = [np.asarray(v).tolist() for v in storable[column]]
        if ANOMALOUS_COLUMN in storable.columns:
            storable[ANOMALOUS_COLUMN] = [[[int(lo), int(hi)] for lo, hi in v]
                                          for v in storable[ANOMALOUS_COLUMN]]
        if "point_score" in storable.columns:
            storable["point_score"] = [np.asarray(v, dtype=float).tolist()
                                       for v in storable["point_score"]]
        storable.to_parquet(path, index=False)

    if diagnostics_path is not None:
        diagnostics_path = Path(diagnostics_path)
        diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
        diagnostics_path.write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")

    return out, votes_frame, diagnostics
