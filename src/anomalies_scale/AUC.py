"""Areas under the ROC and precision-recall curves, computed two ways.

They answer different questions and the module deliberately offers both.

**Ranking AUC**, from :func:`score_points` and :func:`ranking_metrics`. A fixed sliding window
gives every point a continuous score; sorting those scores yields every operating point at once,
so ROC-AUC and PR-AUC come out exactly and in one pass. What this measures is the
*representation* - how well signature distance orders anomalous points above normal ones - with
the widest-first search factored out. It is what separates "the ordering is sound and the
threshold is misplaced" from "no threshold would have helped".

**Operating-point AUC**, from :func:`sweep_operating_points` and :func:`operating_point_auc`.
The real detector is re-run at scaled thresholds and the (FPR, TPR) pairs it traces are
integrated. This measures the *deployed system* - the score, the search and the per-depth
calibration together - and is the more faithful answer to "how does this detector perform across
operating points". It costs one full run of the search per point on the curve.

The second is a legitimate ROC curve and not merely an approximation of the first, because the
flagged sets are nested in the threshold: if an interval is clean at ``t1 < t2`` then its score
is at most ``t1 <= t2``, so at ``t2`` the search declares either it or an ancestor covering it
clean, and boundary extension only grows clean regions further. Raising the threshold can
therefore only remove flags, which is the condition a monotone ROC curve needs.

Three practical differences, which is why the ranking version is the default:

* **Cost.** Each swept point is a complete search over every test stream - about 25 s on FD001
  and 150 s on FD004 - against roughly 20 s for one scoring pass that yields the whole curve.
* **Resolution.** A sweep samples the curve wherever you can afford to; a score gives up to one
  point per distinct value, with no integration error, and the error of a coarse sweep is worst
  at the corners where the area is most sensitive.
* **The threshold is a vector.** Calibration produces one value per dyadic depth, so "sweeping
  the threshold" means choosing a path through a four-dimensional space. Scaling all depths by a
  common factor is the natural choice and the one taken here, but it is a choice; dividing each
  score by its own depth's threshold makes the same decision once and then needs no others.

Scoring a fixed sliding window over every point
-----------------------------------------------

The widest-first search in :mod:`anomalies_scale.stream_scoring` answers a *decision* - which
points are anomalous - and its output is intervals. That is enough for precision, recall and F1,
and not enough for an area, which summarises a detector over every threshold. A binary flag has
already had one threshold applied and discarded the rest; no arithmetic recovers a ranking from
it. Measured on FD001, the area under the three-point curve a binary predictor traces is 0.709 -
balanced accuracy under another name - against 0.877 for the same detector scored continuously.

Deriving the scores from the search's own trace would be cheaper still, and wrong: the search is
adaptive, subdividing only where a test fails, so which intervals get scored depends on the
threshold. An area computed from that trace would move when the threshold moved, which is the
dependence a ranking metric exists to remove. A fixed window asks every point the same question.

Scores are divided by the threshold for the depth they were queried at. Every window here has
the same width and so lands at one depth, making the division a constant that cannot affect the
ranking within a run - but it puts the values on a scale where 1.0 is the firing point, so they
stay comparable between runs, between subsets, and against the notebook's section 8c.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from anomalies_scale.canonical_streams import STREAM_COLUMN, read_canonical
from anomalies_scale.signature_computer import stream_paths
from anomalies_scale.stream_scoring import StreamScorer, resolve_threshold, score_streams

#: Column holding one score per point of the stream, in the frame this module writes.
POINT_SCORE_COLUMN = "point_score"

#: Multipliers applied to every calibrated per-depth threshold when sweeping operating points.
#: Spread around 1.0, which is the calibrated setting itself, so the curve is traced either side
#: of where the detector actually sits rather than only above or below it.
DEFAULT_SCALES = (0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 10.0)


def ranking_metrics(scores, actual):
    """ROC-AUC and PR-AUC over a continuous score - the ordering, not the decision.

    Precision, recall and F1 describe one operating point; these describe the score's ability to
    rank anomalous points above normal ones at *every* operating point, which is what separates a
    representation that cannot distinguish the two from a threshold in the wrong place.

    PR-AUC beside ROC-AUC because the classes are far from balanced - roughly 15% on C-MAPSS -
    and ROC-AUC is optimistic under imbalance. `pr_auc_lift` divides it by the base rate, which
    is what a detector scoring at random achieves, so 1.0 there means no better than chance
    whatever the raw figure looks like.
    """
    actual = np.asarray(actual).astype(bool)
    scores = np.asarray(scores, dtype=float)
    if scores.size != actual.size or not (0 < int(actual.sum()) < actual.size):
        return {}
    if not np.isfinite(scores).all():
        # An infinite score is what the off-manifold test returns and it ranks correctly; a NaN
        # does not, and sklearn would raise on it rather than say why.
        scores = np.nan_to_num(scores, nan=0.0, posinf=np.finfo(float).max)

    from sklearn.metrics import average_precision_score, roc_auc_score

    base = float(actual.mean())
    pr_auc = float(average_precision_score(actual, scores))
    return {"roc_auc": float(roc_auc_score(actual, scores)),
            "pr_auc": pr_auc,
            "pr_auc_lift": pr_auc / base if base else float("nan")}


def resolve_covariance(covariance):
    """The metric, whether it arrived as a `Whitening`, an array, a frame, or a path.

    `load_whitening` keeps the rank-r factor when there is one, so a query is projected into the
    same latent coordinates the index holds. `StreamScorer` wraps anything else.
    """
    if isinstance(covariance, (str, Path)):
        from anomalies_scale.covariance_creation import load_whitening

        return load_whitening(covariance)
    if isinstance(covariance, pd.DataFrame):
        return covariance.to_numpy()
    return covariance


def resolve_normaliser(normaliser):
    if isinstance(normaliser, (str, Path)):
        from anomalies_scale.normalisation import Normaliser

        return Normaliser.load(normaliser)
    return normaliser


def resolve_projection(projection):
    if isinstance(projection, (str, Path)):
        from anomalies_scale.umap_projection import Projection

        return Projection.load(projection)
    return projection


def window_ends(length, window, stride):
    """Last-point index of each window, anchored at its end.

    Anchoring at the end rather than the start is what lets every point take the score of the
    window that closes on it, which is the convention the notebook's sliding curves use and what
    makes a score at index *i* a statement about the history up to *i* rather than the future
    after it.

    A stream shorter than the window yields its own last point, so a short final window is
    scored over what it has rather than dropped - on run-to-failure data that window is the
    failure.
    """
    last = int(length) - 1
    if last < 1:
        return np.zeros(0, dtype=int)
    if last < window:
        return np.asarray([last], dtype=int)
    return np.arange(int(window), last + 1, int(stride), dtype=int)


def spread_to_points(ratios, ends, length):
    """Give every point the score of the first window closing at or after it.

    With ``stride`` 1 that is exactly the window ending on the point. With a coarser stride, or
    over the opening points that no full window closes on, it is the nearest window that does -
    so every point carries a score and none is invented.
    """
    if not len(ends):
        return np.zeros(int(length), dtype=float)
    position = np.clip(np.searchsorted(ends, np.arange(int(length)), side="left"),
                       0, len(ends) - 1)
    return np.asarray(ratios, dtype=float)[position]


def score_points(streams, index, covariance, threshold, window, stride=1, span=None,
                 normaliser=None, trunc=None, signature_projection=None, neighbours=1,
                 output_path=None, show_progress=False):
    """One score per point of every stream, on a fixed sliding window.

    Parameters
    ----------
    window : int
        Increments spanned by the sliding window. Wider is a smoother, easier question, so this
        is a property of the measurement rather than a tuning knob - fix it across a comparison.
    stride : int
        Increments between window ends. 1 scores every point directly; larger strides share a
        score between neighbouring points and cost proportionally less.

    Returns
    -------
    pd.DataFrame
        ``stream`` and `POINT_SCORE_COLUMN` only. The stream data is not copied: `evaluate`
        merges this onto the scored frame, which already carries it.
    """
    frame = read_canonical(streams)
    # `StreamScorer` wants the fitted objects, not paths to them. `score_streams` resolves them
    # on the way in and so must this - the alternative is a scorer that silently treats a
    # filename as a matrix.
    covariance = resolve_covariance(covariance)
    normaliser = resolve_normaliser(normaliser)
    signature_projection = resolve_projection(signature_projection)

    scorer = StreamScorer(index, covariance, threshold, normaliser, trunc, span,
                          None, None, signature_projection, neighbours)

    window, stride = int(window), max(int(stride), 1)
    reference = float(span) if span else None
    rows = []

    for name, path, _ in stream_paths(frame):
        ends = window_ends(len(path), window, stride)
        if not len(ends):
            rows.append({STREAM_COLUMN: name,
                         POINT_SCORE_COLUMN: np.zeros(len(path), dtype=float)})
            continue

        # Every window here is the same width, so they all resolve to one depth and one
        # threshold - which is why this is a single batched search rather than one per depth.
        extent = reference if reference else float(len(path) - 1)
        depth = index.depth_for_width(min(window / extent, 1.0) if extent else 1.0)
        cut = scorer.threshold(depth)

        block = np.vstack([scorer.signature(path, max(end - window, 0), end)[1]
                           for end in ends])
        distances, _ = index.search(block, depth, k=neighbours)
        ratios = np.asarray(distances)[:, -1] / (cut if cut else 1.0)

        rows.append({STREAM_COLUMN: name,
                     POINT_SCORE_COLUMN: spread_to_points(ratios, ends, len(path))})

    out = pd.DataFrame(rows)

    if show_progress:
        total = int(sum(len(v) for v in out[POINT_SCORE_COLUMN]))
        pooled = np.concatenate(list(out[POINT_SCORE_COLUMN])) if len(out) else np.zeros(0)
        print("scored {0:,} point(s) over {1} stream(s) on a {2}-increment window, stride "
              "{3}".format(total, len(out), window, stride))
        if pooled.size:
            print("  score / threshold: median {0:.3f}, p90 {1:.3f}, max {2:.3f}; {3:.2%} of "
                  "points are above 1.0".format(
                      float(np.median(pooled)), float(np.percentile(pooled, 90)),
                      float(pooled.max()), float((pooled > 1.0).mean())))

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        storable = out.copy()
        storable[POINT_SCORE_COLUMN] = [np.asarray(v, dtype=float).tolist()
                                        for v in storable[POINT_SCORE_COLUMN]]
        storable.to_parquet(output_path, index=False)
        if show_progress:
            print("  wrote {0}".format(output_path))

    return out


def sweep_operating_points(streams, index, covariance, threshold, truth, scales=DEFAULT_SCALES,
                           lead_lag=False, output_path=None, show_progress=False, **scoring):
    """Re-run the real detector at scaled thresholds, one row per operating point.

    The calibrated thresholds are a *table* - one value per dyadic depth - so a sweep has to
    choose a path through that space. Every depth is scaled by a common factor here, which keeps
    the relative weighting calibration chose and moves only how permissive the detector is
    overall. 1.0 is the calibrated setting itself and appears in the output, so the deployed
    operating point can be read off the same curve as the hypothetical ones.

    `**scoring` is passed straight to :func:`~anomalies_scale.stream_scoring.score_streams`, so
    this runs the detector exactly as the pipeline does - same search, same span, same
    neighbours - and differs only in the threshold it is handed.
    """
    from anomalies_scale.stream_evaluation import evaluate_predictions

    lookup, _ = resolve_threshold(threshold)
    rows = []

    for scale in sorted(float(s) for s in scales):
        scaled = {int(depth): lookup(depth) * scale for depth in _depths_of(threshold, lookup)}
        scored = score_streams(streams, index, covariance, scaled, show_progress=False,
                               **scoring)
        metrics = evaluate_predictions(scored, truth=truth, lead_lag=lead_lag)

        positives = metrics["tp"] + metrics["fn"]
        negatives = metrics["n_points"] - positives
        rows.append({
            "scale": scale,
            "flagged_fraction": metrics["flagged_fraction"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            # The two coordinates of the ROC plane. Recall is the true positive rate; the false
            # positive rate needs the negative count, which `prf` does not report directly.
            "tpr": metrics["recall"],
            "fpr": metrics["fp"] / negatives if negatives else 0.0,
            "tp": metrics["tp"], "fp": metrics["fp"], "fn": metrics["fn"],
        })
        if show_progress:
            print("  scale {0:>6.2f}  flagged {1:>6.2%}  P {2:.3f}  R {3:.3f}  F1 {4:.3f}"
                  .format(scale, rows[-1]["flagged_fraction"], rows[-1]["precision"],
                          rows[-1]["recall"], rows[-1]["f1"]), flush=True)

    table = pd.DataFrame(rows)
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(output_path, index=False)
        if show_progress:
            print("  wrote {0}".format(output_path))
    return table


def _depths_of(threshold, lookup):
    """Depths the calibrated table covers, so a scale can be applied to each of them."""
    if isinstance(threshold, dict):
        return sorted(int(d) for d in threshold)
    if isinstance(threshold, (int, float, np.integer, np.floating)):
        return [0]
    frame = threshold if isinstance(threshold, pd.DataFrame) else pd.read_csv(Path(threshold))
    return sorted(int(d) for d in frame["depth"])


def operating_point_auc(table):
    """Trapezoidal area under the swept (FPR, TPR) points, anchored at the corners.

    Anchoring is legitimate rather than cosmetic: at an infinite threshold the search flags
    nothing, giving (0, 0), and at zero it flags everything, giving (1, 1). Both are limits of
    the same detector, not invented data.

    Read it as a lower bound. A trapezoid through a handful of sampled points understates a
    concave curve, and the shortfall is worst at the corners where the curve bends most - which
    is exactly why the ranking AUC, which uses every operating point, is the default. The number
    of points behind it is returned so the two are never confused.
    """
    if table is None or not len(table):
        return {}

    points = sorted({(0.0, 0.0), (1.0, 1.0)} |
                    {(float(row.fpr), float(row.tpr)) for row in table.itertuples(index=False)})
    fpr = np.asarray([p[0] for p in points], dtype=float)
    tpr = np.asarray([p[1] for p in points], dtype=float)
    return {"operating_point_auc": float(np.trapz(tpr, fpr)),
            "operating_points": int(len(table))}
