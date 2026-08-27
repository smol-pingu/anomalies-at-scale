"""Score the scorer: plot flagged streams, and measure them against ground truth if there is any.

Consumes what :func:`anomalies_scale.stream_scoring.score_streams` produces - a canonical frame
with an ``anomalous`` column of inclusive ``[lo, hi]`` point indices - and does two things with
it. Plotting needs no labels and is always available. Metrics need labels and are computed only
when they are supplied.

Ground truth in either shape
----------------------------
Labels arrive as one row per stream carrying **either** an ``anomalous`` column of intervals,
in the same convention the scorer emits, **or** a ``label`` column holding a 0/1 array the
length of the stream, which is how SMD and most point-labelled benchmarks ship. Both reduce to
a boolean mask per stream before anything is measured, so the rest of the module never learns
which it was given.

What is measured, and on what axis
----------------------------------
Point-wise precision, recall and F1 over every point of every stream pooled together. Pooling
rather than averaging per stream is deliberate: a fleet where most streams are wholly normal
would otherwise let a stream with two points and no anomalies count for as much as one with
five hundred points and a failure in the middle.

**Point-adjusted** F1 is reported beside the plain figure, never instead of it. The adjustment -
if any point of a true segment is flagged, credit the whole segment - is standard in the time
series anomaly literature and is extremely flattering; a detector that flags one point at random
inside each anomaly scores near 1.0. It is included because papers quote it, and paired with the
unadjusted number so the gap between them is visible rather than hidden.

The degenerate baselines are reported too. "Flag everything" is the floor any detector has to
beat, and on data with a 15% anomaly rate it already scores an F1 of 0.26 - which is why a bare
F1 with nothing to compare it against says very little.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from anomalies_scale.canonical_streams import (
    STREAM_COLUMN, TIME_COLUMN, VALUES_COLUMN, as_matrix)
from anomalies_scale.stream_scoring import ANOMALOUS_COLUMN
from anomalies_scale.AUC import POINT_SCORE_COLUMN, ranking_metrics
from anomalies_scale.windowing import reassemble

#: Column a per-point ground truth uses.
LABEL_COLUMN = "label"


def collapse_lead_lag(frame):
    """Map lead-lag streams back onto the observations they were built from.

    The inverse of :func:`anomalies_scale.preprocessing.lead_lag_columns`. That transform takes
    an *n*-point stream to ``2n - 1`` points by advancing the lead and the lag alternately, so
    observation *i* occupies indices ``{2i - 1, 2i}`` - and ``{0}`` for the first, which has no
    preceding lag step. Even indices hold the lead at rest, so ``[::2]`` recovers the original
    sequence exactly.

    This has to run before anything is measured. Labels are stated per observation, a lead-lag
    index is not an observation, and a stream twice its expected length would either raise on
    the mismatch or - worse, if labels were regenerated at the doubled length - quietly redefine
    the horizon to half as many cycles.

    A point is called anomalous if *either* of its two lead-lag indices was flagged. The two are
    the same observation seen from its lead and its lag step; requiring both would discard every
    flag that landed on one side of the interleave.
    """
    out = []
    for row in frame.itertuples(index=False):
        values = as_matrix(getattr(row, VALUES_COLUMN))
        doubled = len(values)
        length = (doubled + 1) // 2

        record = {
            STREAM_COLUMN: str(getattr(row, STREAM_COLUMN)),
            # The lead channels only - the lag half is a delayed copy of them, not new data.
            VALUES_COLUMN: values[::2, :values.shape[1] // 2],
        }
        if TIME_COLUMN in frame.columns:
            record[TIME_COLUMN] = np.asarray(getattr(row, TIME_COLUMN), dtype=float)[::2]

        if ANOMALOUS_COLUMN in frame.columns:
            wide = intervals_to_mask(getattr(row, ANOMALOUS_COLUMN), doubled)
            mask = np.zeros(length, dtype=bool)
            mask[0] = wide[0]
            if length > 1:
                positions = np.arange(1, length)
                mask[1:] = wide[2 * positions - 1] | wide[2 * positions]
            record[ANOMALOUS_COLUMN] = [[int(lo), int(hi - 1)] for lo, hi in segments(mask)]
            record["n_anomalous_intervals"] = len(record[ANOMALOUS_COLUMN])
            record["n_anomalous_points"] = int(mask.sum())
            record["anomalous_fraction"] = float(mask.mean())
        if "max_score" in frame.columns:
            record["max_score"] = float(getattr(row, "max_score"))
        out.append(record)

    return pd.DataFrame(out)


def read_scored(scored, lead_lag=False, point_scores=None):
    """Read a scored frame and undo the transforms that make its indices non-physical.

    Windowing is invisible from here on. Ground truth is stated per source stream - nobody
    labelled a window - and metrics taken per window would count every shared boundary point
    twice and split an anomaly straddling a window edge into two segments. `reassemble` returns
    an unwindowed frame untouched, so this is safe to apply either way.

    `lead_lag` says the streams were interleaved with their own past in preprocessing, doubling
    their length. That cannot be detected from the frame - a doubled stream looks like an
    ordinary one - so it is passed down from the configuration that asked for it.

    `point_scores` is the continuous score per point that `point_scoring` produces, merged in
    *before* reassembly so it is folded back into source coordinates alongside the values it
    describes rather than being aligned to windows afterwards.
    """
    frame = scored if isinstance(scored, pd.DataFrame) else pd.read_parquet(Path(scored))

    if point_scores is not None:
        extra = (point_scores if isinstance(point_scores, pd.DataFrame)
                 else pd.read_parquet(Path(point_scores)))
        frame = frame.merge(extra[[STREAM_COLUMN, POINT_SCORE_COLUMN]],
                            on=STREAM_COLUMN, how="left")

    frame = reassemble(frame, anomalous_column=ANOMALOUS_COLUMN,
                       score_column=POINT_SCORE_COLUMN)
    return collapse_lead_lag(frame) if lead_lag else frame


def intervals_to_mask(intervals, length):
    """Rasterise inclusive ``[lo, hi]`` pairs onto a boolean mask of `length` points."""
    mask = np.zeros(int(length), dtype=bool)
    for pair in intervals if intervals is not None else []:
        lo, hi = int(pair[0]), int(pair[1])
        mask[max(lo, 0):min(hi, length - 1) + 1] = True
    return mask


def segments(mask):
    """Contiguous True runs of a mask, as half-open ``(lo, hi)`` pairs."""
    edges = np.diff(np.concatenate([[0], np.asarray(mask).astype(int), [0]]))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def truth_masks(truth, lengths):
    """Reduce a ground-truth frame to ``{stream: boolean mask}``, whichever shape it came in.

    `lengths` says how long each stream is, which an interval-shaped truth cannot know by
    itself. A stream named in the labels but absent from the predictions is ignored with the
    others counted, rather than silently shifting the alignment.
    """
    frame = truth if isinstance(truth, pd.DataFrame) else pd.read_parquet(Path(truth))
    if STREAM_COLUMN not in frame.columns:
        raise ValueError("ground truth needs a {0!r} column; got {1}".format(
            STREAM_COLUMN, list(frame.columns)[:8]))

    has_intervals = ANOMALOUS_COLUMN in frame.columns
    has_points = LABEL_COLUMN in frame.columns
    if not (has_intervals or has_points):
        raise ValueError(
            "ground truth must carry either {0!r} (intervals) or {1!r} (per-point); got "
            "{2}".format(ANOMALOUS_COLUMN, LABEL_COLUMN, list(frame.columns)[:8]))

    masks = {}
    for row in frame.itertuples(index=False):
        name = str(getattr(row, STREAM_COLUMN))
        if name not in lengths:
            continue
        if has_points:
            values = np.asarray(getattr(row, LABEL_COLUMN)).astype(bool)
            if len(values) != lengths[name]:
                raise ValueError(
                    "stream {0!r} has {1} label(s) for {2} point(s)".format(
                        name, len(values), lengths[name]))
            masks[name] = values
        else:
            masks[name] = intervals_to_mask(getattr(row, ANOMALOUS_COLUMN), lengths[name])
    return masks, "per-point" if has_points else "intervals"


def prf(predicted, actual):
    """Precision, recall and F1 of two boolean masks."""
    tp = int((predicted & actual).sum())
    fp = int((predicted & ~actual).sum())
    fn = int((~predicted & actual).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1,
            "tp": tp, "fp": fp, "fn": fn}


def point_adjust(predicted, actual):
    """Expand each true segment to fully flagged if any point inside it was flagged.

    The convention most of the time series anomaly literature reports, and a very generous one:
    a detector that hits one point at random inside every anomaly scores close to 1.0. Reported
    alongside the unadjusted figure, never in place of it.
    """
    adjusted = np.asarray(predicted).copy()
    for lo, hi in segments(actual):
        if adjusted[lo:hi].any():
            adjusted[lo:hi] = True
    return adjusted


#: Exathlon's four levels of anomaly-detection functionality, from basic to advanced.
AD_LEVELS = ("ad1", "ad2", "ad3", "ad4")


def _overlap(first, second):
    """Intersection of two half-open ranges, or None where they do not meet."""
    lo, hi = max(first[0], second[0]), min(first[1], second[1])
    return (lo, hi) if hi > lo else None


def _covered_fraction(span, overlaps):
    """How much of `span` the overlaps cover, as a fraction of its length."""
    length = span[1] - span[0]
    if length <= 0:
        return 0.0
    return float(sum(hi - lo for lo, hi in overlaps)) / length


def _range_recall(real, predicted, level):
    """Recall credit for one true anomaly range, at one AD level.

    The four levels are cumulative, each adding a requirement to the one before, which is what
    makes the scores fall monotonically - Exathlon's own design guarantee, and a property worth
    checking rather than assuming.

    ``ad1``  flagged at all, or not.
    ``ad2``  how much of the range was covered.
    ``ad3``  that, reduced by how late the first flag arrived - the detection latency AD3 names.
    ``ad4``  that, but zero unless exactly one predicted range covers it.
    """
    overlaps = [o for o in (_overlap(real, p) for p in predicted) if o is not None]
    if not overlaps:
        return 0.0
    if level == "ad1":
        return 1.0

    reward = _covered_fraction(real, overlaps)
    if level == "ad2":
        return reward

    # Latency measured from the range's own start, in units of its own length, so a late flag on
    # a long anomaly is penalised less than the same delay on a short one.
    length = real[1] - real[0]
    latency = (min(lo for lo, _ in overlaps) - real[0]) / length if length else 0.0
    reward *= max(0.0, 1.0 - latency)
    if level == "ad3":
        return reward

    return reward if len(overlaps) == 1 else 0.0


def _range_precision(predicted, real, level):
    """Precision credit for one predicted range.

    Coverage quality is a property of the *anomaly*, not of the prediction, so AD1 and AD3 add
    nothing here - the paper says so explicitly, and its worked example has all four levels
    considering the same predicted subranges. Only size (AD2) and number (AD4) apply.
    """
    overlaps = [o for o in (_overlap(predicted, r) for r in real) if o is not None]
    if not overlaps:
        return 0.0
    reward = _covered_fraction(predicted, overlaps)
    if level == "ad4" and len(overlaps) != 1:
        return 0.0
    return reward


def range_metrics(predicted_masks, actual_masks):
    """Range-based precision, recall and F-score at each of Exathlon's four AD levels.

    Point-wise metrics count a detector by how many *points* it got right, which rewards
    covering one long anomaly and says nothing about whether the other ninety-nine were found at
    all. Range-based metrics score each anomaly *range* once and average, so every event counts
    equally however long it happens to be - the framework of Tatbul et al. (NeurIPS 2018), which
    Exathlon parameterises into four levels of increasing strictness (VLDB 2021, section 4).

    Ranges are found per stream and never across a stream boundary: two engines whose records
    happen to abut, one ending anomalous and the next beginning so, are two events and not one.

    The exact parameter settings Exathlon uses are deferred to its technical report, which is not
    public alongside the paper. This implements the semantics the paper states - existence,
    coverage, latency, exactly-once - rather than reproducing those settings bit for bit, so
    treat the numbers as faithful to the definitions rather than comparable digit-by-digit with
    the paper's own tables.
    """
    scored = {level: {"recall": [], "precision": []} for level in AD_LEVELS}

    for predicted, actual in zip(predicted_masks, actual_masks):
        real = segments(actual)
        found = segments(predicted)
        for level in AD_LEVELS:
            scored[level]["recall"].extend(
                _range_recall(r, found, level) for r in real)
            scored[level]["precision"].extend(
                _range_precision(p, real, level) for p in found)

    out = {}
    for level in AD_LEVELS:
        recalls, precisions = scored[level]["recall"], scored[level]["precision"]
        recall = float(np.mean(recalls)) if recalls else 0.0
        precision = float(np.mean(precisions)) if precisions else 0.0
        out["{0}_recall".format(level)] = recall
        out["{0}_precision".format(level)] = precision
        out["{0}_f1".format(level)] = (2 * precision * recall / (precision + recall)
                                       if precision + recall else 0.0)
    out["n_predicted_ranges"] = len(scored["ad2"]["precision"])
    return out


def evaluate_predictions(scored, truth=None, lead_lag=False, point_scores=None,
                         show_progress=False):
    """Measure scored streams against ground truth, if any is given.

    Returns
    -------
    dict
        Pooled point-wise metrics, the point-adjusted variant, the degenerate baselines, and
        the label rate. Empty of metrics when `truth` is None. Adds ROC-AUC and PR-AUC when
        `point_scores` supplies a continuous score to rank by.
    """
    frame = read_scored(scored, lead_lag, point_scores)
    lengths = {str(getattr(row, STREAM_COLUMN)): len(as_matrix(getattr(row, VALUES_COLUMN)))
               for row in frame.itertuples(index=False)}

    # Kept per stream as well as pooled. Point-wise metrics want one long array; range metrics
    # must not see two adjacent streams as one event, so they need the boundaries.
    per_stream = [
        intervals_to_mask(getattr(row, ANOMALOUS_COLUMN), lengths[str(getattr(row, STREAM_COLUMN))])
        for row in frame.itertuples(index=False)]
    predicted = (np.concatenate(per_stream) if per_stream else np.zeros(0, dtype=bool))

    summary = {"n_streams": len(frame), "n_points": int(predicted.size),
               "flagged_fraction": float(predicted.mean()) if predicted.size else 0.0}
    if truth is None:
        return summary

    masks, shape = truth_masks(truth, lengths)
    missing = [name for name in lengths if name not in masks]
    if missing:
        raise ValueError(
            "{0} scored stream(s) have no ground truth, so a pooled metric would count them "
            "as wholly normal: {1}".format(len(missing), sorted(missing)[:5]))

    actual_per_stream = [masks[str(getattr(row, STREAM_COLUMN))]
                         for row in frame.itertuples(index=False)]
    actual = np.concatenate(actual_per_stream)

    summary.update(prf(predicted, actual))
    summary.update(range_metrics(per_stream, actual_per_stream))
    summary["truth_shape"] = shape
    summary["anomaly_rate"] = float(actual.mean())
    summary["n_true_segments"] = len(segments(actual))

    adjusted = prf(point_adjust(predicted, actual), actual)
    summary["adjusted_precision"] = adjusted["precision"]
    summary["adjusted_recall"] = adjusted["recall"]
    summary["adjusted_f1"] = adjusted["f1"]

    everything = prf(np.ones_like(actual), actual)
    summary["baseline_flag_all_f1"] = everything["f1"]
    summary["beats_baseline"] = bool(summary["f1"] > everything["f1"])

    if POINT_SCORE_COLUMN in frame.columns:
        summary.update(ranking_metrics(
            np.concatenate([np.asarray(getattr(row, POINT_SCORE_COLUMN), dtype=float)
                            for row in frame.itertuples(index=False)]), actual))

    if show_progress:
        print("{n_points:,} point(s) over {n_streams} stream(s); truth given as {truth_shape}, "
              "{anomaly_rate:.2%} anomalous in {n_true_segments} segment(s)".format(**summary))
        print("  P {precision:.3f}  R {recall:.3f}  F1 {f1:.3f}   |   adjusted F1 "
              "{adjusted_f1:.3f}   |   flagged {flagged_fraction:.2%}".format(**summary))
        print("  flag-everything F1 is {baseline_flag_all_f1:.3f}; this detector {0} "
              "it".format("beats" if summary["beats_baseline"] else "DOES NOT beat", **summary))
        if "roc_auc" in summary:
            print("  ROC-AUC {roc_auc:.3f}   PR-AUC {pr_auc:.3f} against a base rate of "
                  "{anomaly_rate:.3f} ({pr_auc_lift:.1f}x lift)".format(**summary))
        print("  range-based, over {0} true and {1} predicted range(s):".format(
            summary["n_true_segments"], summary["n_predicted_ranges"]))
        print("    {0:<6}{1:>10}{2:>10}{3:>10}   {4}".format(
            "level", "precision", "recall", "F1", "adds"))
        for level, adds in zip(AD_LEVELS, ("existence", "coverage", "latency", "exactly-once")):
            print("    {0:<6}{1:>10.3f}{2:>10.3f}{3:>10.3f}   {4}".format(
                level.upper(), summary["{0}_precision".format(level)],
                summary["{0}_recall".format(level)], summary["{0}_f1".format(level)], adds))
    return summary


def plot_scored_streams(scored, truth=None, output_dir=None, max_plots=12, max_channels=3,
                        order="most_flagged", lead_lag=False, show_progress=False):
    """Draw streams with their flagged intervals shaded, and truth beneath if given.

    `order` decides which streams are drawn when there are more than `max_plots`:
    ``'most_flagged'`` takes the ones the detector had most to say about, ``'first'`` takes them
    in order. Most-flagged by default because a fleet is mostly quiet and drawing the quiet
    ones first shows nothing.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    frame = read_scored(scored, lead_lag)
    lengths = {str(getattr(row, STREAM_COLUMN)): len(as_matrix(getattr(row, VALUES_COLUMN)))
               for row in frame.itertuples(index=False)}
    masks = truth_masks(truth, lengths)[0] if truth is not None else {}

    chosen = frame
    if order == "most_flagged" and "n_anomalous_points" in frame.columns:
        chosen = frame.sort_values("n_anomalous_points", ascending=False)
    chosen = chosen.head(max_plots)

    written = []
    for row in chosen.itertuples(index=False):
        name = str(getattr(row, STREAM_COLUMN))
        values = as_matrix(getattr(row, VALUES_COLUMN))
        intervals = getattr(row, ANOMALOUS_COLUMN, [])

        # The channels that move most, since a constant one draws a flat line and says nothing.
        spread = values.std(axis=0)
        channels = np.argsort(spread)[::-1][:max_channels]
        channels = [c for c in channels if spread[c] > 0] or [0]

        bands = 1 + (name in masks)
        fig, axes = plt.subplots(len(channels) + bands, 1, figsize=(13, 2 + 1.6 * len(channels)),
                                 sharex=True,
                                 gridspec_kw={"height_ratios": [3] * len(channels) + [1] * bands})
        axes = np.atleast_1d(axes)

        for ax, channel in zip(axes, channels):
            ax.plot(values[:, channel], lw=0.7, color="#1f77b4")
            ax.set_ylabel("ch {0}".format(channel), fontsize=8)
            ax.margins(x=0)

        strips = [(intervals_to_mask(intervals, lengths[name]), "detected", "darkorange")]
        if name in masks:
            strips.insert(0, (masks[name], "truth", "crimson"))
        for ax, (mask, label, colour) in zip(axes[len(channels):], strips):
            for lo, hi in segments(mask):
                ax.axvspan(lo, hi - 1, color=colour, alpha=0.75, lw=0)
            ax.set_yticks([])
            ax.set_ylabel(label, fontsize=8, rotation=0, ha="right", va="center")
            ax.set_xlim(0, lengths[name] - 1)

        axes[-1].set_xlabel("point index")
        # `len(intervals or [])` would work on a list and raises on the numpy array parquet
        # hands back, since truth-testing a multi-element array is ambiguous.
        n_intervals = 0 if intervals is None else len(intervals)
        fig.suptitle("{0} - {1} interval(s) flagged over {2} point(s)".format(
            name, n_intervals, lengths[name]), fontsize=10)
        plt.tight_layout()

        if output_dir is not None:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            destination = output_dir / "{0}.png".format(str(name).replace("/", "_"))
            fig.savefig(destination, dpi=110, bbox_inches="tight")
            written.append(destination)
        plt.close(fig)

    if show_progress:
        print("drew {0} stream(s){1}".format(
            len(written) or len(chosen),
            " to {0}".format(output_dir) if output_dir else ""))
    return written
