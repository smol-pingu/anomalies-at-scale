"""Turn scored intervals into a verdict per interval, a summary per stream, and figures.

The scoring stage reports the *complement* of the clean blocks it found, which is the
procedure's definition of anomalous. This stage says what that means: where each flagged
interval sits, how far past its own threshold it went, and whether it survives being tested
as a single interval rather than only failing at the scales the search bisected to.

Verdicts
--------
``anomalous``
    The interval exceeds the threshold calibrated for its own width when tested directly.

``inconclusive``
    The interval is part of the complement - the search found no clean block covering it -
    yet it passes its threshold when measured whole. That is not a contradiction. Signature
    distance is not monotone in interval width, so a region can look wrong at every scale the
    search examined and unremarkable when taken in one piece, which is what a diffuse
    departure from normality looks like. Reported, not silently dropped, and not counted as a
    detection either.

Severity is reported as ``ratio``, the distance divided by its own width's threshold. Raw
distances are not comparable between widths - a level-*k* signature term scales like
(increment) ** k - so the ratio is the only figure that can be ranked across a whole run.

Figures
-------
One per stream, value channels stacked, flagged regions shaded. Streams with nothing flagged
are skipped by default: on a run of any size they are the overwhelming majority, and a
directory of flat lines buries the few plots worth looking at.
"""

from __future__ import annotations

import argparse
import base64
from datetime import datetime
from pathlib import Path

import matplotlib

# A pipeline stage draws to disk and never to a screen; choosing the backend explicitly stops
# it from depending on whatever display happens to exist where it runs.
matplotlib.use("Agg")

import matplotlib.pyplot as plt          # noqa: E402  (must follow the backend choice)
import numpy as np                       # noqa: E402
import pandas as pd                      # noqa: E402

from anomalies_scale.covariance_creation import read_corpus          # noqa: E402
from anomalies_scale.canonical_streams import (                      # noqa: E402
    expand_paths, iter_streams)
from anomalies_scale.signature_computer import write_corpus          # noqa: E402

#: Verdicts an interval can receive.
ANOMALOUS = "anomalous"
INCONCLUSIVE = "inconclusive"


def classify_intervals(scores):
    """Attach a verdict and a rank to each scored interval.

    Ranking is by `ratio` rather than raw distance, since distances at different interval
    widths are measured against different thresholds and cannot be ordered against each other.
    """
    frame = pd.DataFrame(scores).copy()
    if frame.empty:
        frame["verdict"] = pd.Series(dtype=object)
        frame["rank"] = pd.Series(dtype=int)
        return frame

    frame["verdict"] = np.where(frame["exceeds_threshold"], ANOMALOUS, INCONCLUSIVE)
    frame["rank"] = frame["ratio"].rank(ascending=False, method="min").astype(int)
    return frame.sort_values("rank").reset_index(drop=True)


def summarise_streams(intervals, stream_lengths=None):
    """One row per stream: how much was flagged, how badly, and over what.

    `stream_lengths` lets the flagged fraction be reported against the whole stream rather
    than only against the part that was flagged, which is the figure that says whether a
    detection is a localised event or a claim about the entire series.
    """
    if intervals.empty:
        return pd.DataFrame(columns=["stream", "n_intervals", "n_anomalous",
                                     "n_inconclusive", "points_flagged", "stream_points",
                                     "fraction_flagged", "max_ratio"])

    rows = []
    for name, group in intervals.groupby("stream", sort=True):
        anomalous = group[group["verdict"] == ANOMALOUS]
        points = int(anomalous["n_points"].sum())
        length = None if stream_lengths is None else stream_lengths.get(name)
        rows.append({
            "stream": name,
            "n_intervals": len(group),
            "n_anomalous": len(anomalous),
            "n_inconclusive": int((group["verdict"] == INCONCLUSIVE).sum()),
            "points_flagged": points,
            "stream_points": length,
            "fraction_flagged": (points / length) if length else np.nan,
            "max_ratio": float(group["ratio"].max()),
        })
    return pd.DataFrame(rows).sort_values("max_ratio", ascending=False).reset_index(drop=True)


def plot_stream(path, name, intervals, max_channels=5, ax=None):
    """Draw one stream with its flagged regions shaded.

    Anomalous intervals are shaded solidly; inconclusive ones are outlined instead, so the
    distinction survives into the picture rather than only living in the table.
    """
    path = np.asarray(path, dtype=float)
    time, values = path[:, 0], path[:, 1:]
    channels = min(max_channels, values.shape[1])

    if ax is None:
        figure, axes = plt.subplots(channels, 1, figsize=(13, 1.9 * channels + 0.8),
                                    sharex=True, squeeze=False)
        axes = axes[:, 0]
    else:
        figure, axes = ax.figure, [ax]
        channels = 1

    for row in range(channels):
        panel = axes[row]
        panel.plot(time, values[:, row], lw=0.7, color="#1f77b4")
        for interval in intervals.itertuples():
            if interval.verdict == ANOMALOUS:
                panel.axvspan(interval.start_time, interval.end_time,
                              facecolor="crimson", edgecolor="none", alpha=0.22, lw=0)
            else:
                # Outlined rather than filled: reported, but not counted as a detection.
                panel.axvspan(interval.start_time, interval.end_time,
                              facecolor="none", edgecolor="goldenrod", lw=1.1, ls="--")
        panel.set_ylabel("ch {0}".format(row), fontsize=8)
        panel.margins(x=0)
        panel.spines[["top", "right"]].set_visible(False)

    n_anomalous = int((intervals["verdict"] == ANOMALOUS).sum()) if len(intervals) else 0
    axes[0].set_title(
        "{0} - {1} anomalous interval(s){2}".format(
            name, n_anomalous,
            ", {0} inconclusive".format(len(intervals) - n_anomalous)
            if len(intervals) > n_anomalous else ""),
        fontsize=10)
    axes[-1].set_xlabel("time")
    figure.tight_layout()
    return figure


REPORT_STYLE = """
body { font-family: -apple-system, Segoe UI, Roboto, sans-serif; max-width: 1100px;
       margin: 2rem auto; padding: 0 1.5rem; line-height: 1.55; color: #1a1a1a; }
h1 { border-bottom: 2px solid #1f77b4; padding-bottom: .3rem; }
h2 { margin-top: 2.5rem; border-bottom: 1px solid #ddd; padding-bottom: .2rem; }
table { border-collapse: collapse; margin: 1rem 0; font-size: .9rem; }
th, td { border: 1px solid #ddd; padding: .35rem .6rem; text-align: right; }
th { background: #f4f6f8; text-align: left; }
td:first-child, th:first-child { text-align: left; }
.figure { margin: 1.5rem 0; }
.figure img { max-width: 100%; border: 1px solid #eee; }
.meta { color: #666; font-size: .9rem; }
.empty { color: #888; font-style: italic; }
"""


def _table(frame, limit=50):
    """Render a frame as HTML, saying so when it has been cut short."""
    if frame is None or not len(frame):
        return '<p class="empty">nothing to show</p>'
    shown = frame.head(limit)
    html = shown.to_html(index=False, float_format=lambda v: "{0:.4g}".format(v),
                         border=0, na_rep="")
    if len(frame) > limit:
        html += '<p class="meta">showing {0} of {1} rows</p>'.format(limit, len(frame))
    return html


def _figures(figures_dir, limit=25):
    """Inline every figure as a data URI.

    Inlined rather than linked so the report is one portable file. A report that only renders
    from the directory it was built in is not much of a report - it cannot be attached to
    anything or kept after a `clean`.
    """
    directory = Path(figures_dir) if figures_dir else None
    if directory is None or not directory.is_dir():
        return '<p class="empty">no figures were produced</p>'

    images = sorted(directory.glob("*.png"))
    if not images:
        return '<p class="empty">no figures were produced</p>'

    blocks = []
    for image in images[:limit]:
        encoded = base64.b64encode(image.read_bytes()).decode("ascii")
        blocks.append(
            '<div class="figure"><p class="meta">{0}</p>'
            '<img src="data:image/png;base64,{1}" alt="{0}"></div>'.format(
                image.stem, encoded))
    if len(images) > limit:
        blocks.append('<p class="meta">{0} further figure(s) in {1}</p>'.format(
            len(images) - limit, directory))
    return "\n".join(blocks)


#: Point-wise metrics, in the order they are worth reading rather than alphabetically.
POINTWISE_KEYS = ("precision", "recall", "f1", "tp", "fp", "fn",
                  "adjusted_precision", "adjusted_recall", "adjusted_f1",
                  "flagged_fraction", "anomaly_rate", "n_true_segments", "n_predicted_ranges",
                  "truth_shape", "baseline_flag_all_f1", "beats_baseline")

#: Threshold-free metrics. Kept apart from the point-wise ones because they answer a different
#: question - how well the score *orders* points, rather than how one operating point performed.
RANKING_KEYS = ("roc_auc", "pr_auc", "pr_auc_lift", "operating_point_auc", "operating_points")

#: Exathlon's four levels, and what each adds to the one before it.
AD_LADDER = (("ad1", "flagged at all"), ("ad2", "+ how much was covered"),
             ("ad3", "+ how late the first flag was"), ("ad4", "+ reported exactly once"))

#: Per-stream columns worth tabulating. The scored frame also carries `Time`, `Stream` and
#: `anomalous`, which are arrays per row and would render as pages of numbers.
PER_STREAM_KEYS = ("stream", "n_anomalous_intervals", "n_anomalous_points",
                   "anomalous_fraction", "max_score")


def _ordered_table(metrics, keys):
    """Selected metrics as a two-column frame, in the given order rather than sorted."""
    rows = [(key, metrics[key]) for key in keys if key in metrics]
    return pd.DataFrame(rows, columns=["quantity", "value"]) if rows else None


def _ad_table(metrics):
    """The four AD levels as a ladder, which is the only order they make sense in.

    Sorted alphabetically - which is what a generic dump of the summary does - the four levels
    scatter among the other metrics and the cumulative structure that makes them worth reporting
    is invisible.
    """
    rows = [{"level": level.upper(), "adds": adds,
             "precision": metrics["{0}_precision".format(level)],
             "recall": metrics["{0}_recall".format(level)],
             "F1": metrics["{0}_f1".format(level)]}
            for level, adds in AD_LADDER if "{0}_f1".format(level) in metrics]
    return pd.DataFrame(rows) if rows else None


def _flatten_rows(block, prefix=""):
    """Nested dicts as flat ``(section.key, value)`` pairs.

    Separate from the frame it becomes because the recursion has to accumulate *rows*: extending
    a list with a DataFrame iterates its column names, not its contents.
    """
    rows = []
    for key, value in (block or {}).items():
        name = "{0}{1}".format(prefix, key)
        if isinstance(value, dict):
            rows.extend(_flatten_rows(value, "{0}.".format(name)))
        else:
            rows.append((name, value))
    return rows


def _flatten(block):
    """Nested dicts as a `section.key` table, so a structure renders as rows not a repr."""
    rows = _flatten_rows(block)
    return pd.DataFrame(rows, columns=["quantity", "value"]) if rows else None


def build_report(output_path, dataset, metrics=None, per_stream=None, thresholds=None,
                 figures_dir=None, config=None, show_progress=False):
    """Assemble a run's artifacts into one self-contained HTML document.

    Reads what the earlier stages wrote rather than recomputing any of it, so the report can
    never disagree with the files it describes. Every input is optional: a run that stopped
    early still produces a report saying what it did have, which is more use than no report.

    The configuration is included because the numbers are meaningless without it - a threshold
    of 1994 says nothing unless the truncation level, granularity and retained variance that
    produced it are on the same page.
    """
    def read(path):
        if path is None or not Path(path).exists():
            return None
        path = Path(path)
        if path.suffix.lower() == ".csv":
            return pd.read_csv(path)
        if path.suffix.lower() == ".json":
            # The metrics summary is a flat dict of scalars, which reads better as one
            # quantity per row than as a single very wide row.
            import json

            data = json.loads(path.read_text(encoding="utf-8"))
            # Returned as the dict it is. The caller decides which metrics belong in which
            # section; dumping every key into one alphabetised table is what used to bury the
            # AD ladder among unrelated quantities.
            return data if isinstance(data, dict) else pd.DataFrame(data)
        return read_corpus(path)

    measured = read(metrics) or {}
    thresholds_frame = read(thresholds)

    # Per-stream rows come from the scored frame, and the pooled metrics from the summary. They
    # were previously the same input under one heading, which put dataset-level metrics under a
    # "per-stream" title and left the actual per-stream table unrendered.
    scored_frame = read(per_stream)
    if scored_frame is not None:
        keep = [c for c in PER_STREAM_KEYS if c in scored_frame.columns]
        scored_frame = (scored_frame[keep].sort_values("anomalous_fraction", ascending=False)
                        if keep and "anomalous_fraction" in keep
                        else (scored_frame[keep] if keep else None))

    headline = [("streams scored", measured.get("n_streams")),
                ("points scored", measured.get("n_points")),
                ("flagged", measured.get("flagged_fraction"))]
    if "f1" in measured:
        headline += [("point-wise F1", measured["f1"]),
                     ("AD1 recall (anomalies found at all)", measured.get("ad1_recall")),
                     ("ROC-AUC", measured.get("roc_auc")),
                     ("PR-AUC", measured.get("pr_auc"))]
    else:
        headline.append(("detector evaluation",
                         "off - evaluate.metrics is false or no labels were given"))
    headline = [(label, value) for label, value in headline if value is not None]

    config_rows = pd.DataFrame(
        [{"section": section, "setting": key, "value": value}
         for section, block in sorted((config or {}).items())
         for key, value in (sorted(block.items()) if isinstance(block, dict)
                            else [("", block)])])

    document = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>{dataset} - anomaly detection report</title>
<style>{style}</style></head><body>
<h1>{dataset}</h1>
<p class="meta">generated {when}</p>

<h2>Headline</h2>
{headline}

<h2>Detector evaluation</h2>
<p class="meta">How the detector scored against ground truth. Computed only when
<code>evaluate.metrics</code> is enabled and labels were supplied.</p>

<h3>Point-wise</h3>
<p class="meta">Every point counted equally, so a long anomaly weighs more than a short one.
The adjusted figures expand each true segment to fully flagged if any point inside it was
caught - generous, and reported beside the unadjusted ones rather than instead of them.</p>
{pointwise}

<h3>Range-based, at Exathlon's four levels</h3>
<p class="meta">Each anomaly scored once and averaged, so every event counts equally however
long it is. The levels are cumulative and therefore fall monotonically.</p>
{ad_levels}

<h3>Ranking</h3>
<p class="meta">Threshold-free. <code>roc_auc</code> and <code>pr_auc</code> rank the
sliding-window score and measure the representation; <code>operating_point_auc</code> integrates
the real detector re-run at scaled thresholds and measures the deployed system.</p>
{ranking}

<h2>What the run cost</h2>
{throughput}

<h2>Calibrated thresholds</h2>
{thresholds}

<h2>Per-stream summary</h2>
<p class="meta">One row per scored stream, worst first.</p>
{per_stream}

<h2>Figures</h2>
{figures}

<h2>Configuration</h2>
{config}
</body></html>
""".format(
        dataset=dataset,
        style=REPORT_STYLE,
        when=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        headline=_table(pd.DataFrame(headline, columns=["quantity", "value"])),
        pointwise=_table(_ordered_table(measured, POINTWISE_KEYS)),
        ad_levels=_table(_ad_table(measured)),
        ranking=_table(_ordered_table(measured, RANKING_KEYS)),
        throughput=_table(_flatten(measured.get("throughput")), limit=100),
        thresholds=_table(thresholds_frame),
        per_stream=_table(scored_frame),
        figures=_figures(figures_dir),
        config=_table(config_rows, limit=200),
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(document, encoding="utf-8")
    if show_progress:
        print("wrote {0} ({1:.0f} KB)".format(
            output_path, output_path.stat().st_size / 1024))
    return output_path


def evaluate_scores(scores, streams, output_path=None, summary_path=None,
                    figures_dir=None, plot_clean=False, max_plots=25, max_channels=5,
                    time_col=None, value_cols=None, stream_col=None, show_progress=False):
    """Report the scored intervals and draw the streams they came from.

    Parameters
    ----------
    scores : str or Path or pd.DataFrame
        Output of the scoring stage.
    streams : iterable of str or Path
        The stream files that were scored, needed to draw them.
    plot_clean : bool
        Also draw streams with nothing flagged. Off by default - on a run of any size they
        are the overwhelming majority.
    max_plots : int
        Cap on figures drawn, most severe first. ``None`` for no cap.

    Returns
    -------
    (pd.DataFrame, pd.DataFrame)
        The per-interval report and the per-stream summary.
    """
    intervals = classify_intervals(read_corpus(scores))

    paths, lengths = {}, {}
    for file in [Path(f) for f in streams]:
        for name, path in iter_streams(file, stream_col=stream_col, time_col=time_col,
                                       value_cols=value_cols):
            paths[name] = path
            lengths[name] = int(path.shape[0])

    missing = set(intervals["stream"]) - set(paths) if len(intervals) else set()
    if missing:
        raise ValueError(
            "scored intervals reference {0} stream(s) not found in the files given: {1}. "
            "The streams scored and the streams being reported on must match."
            .format(len(missing), sorted(missing)[:5]))

    summary = summarise_streams(intervals, lengths)

    if show_progress:
        counts = intervals["verdict"].value_counts() if len(intervals) else {}
        print("{0} flagged interval(s) across {1} stream(s): {2} anomalous, {3} "
              "inconclusive".format(
                  len(intervals), summary.shape[0],
                  int(counts.get(ANOMALOUS, 0)), int(counts.get(INCONCLUSIVE, 0))))
        if len(summary):
            print()
            print(summary.to_string(index=False))

    if output_path is not None:
        write_corpus(intervals, output_path)
    if summary_path is not None:
        write_corpus(summary, summary_path)

    written = []
    if figures_dir is not None:
        figures_dir = Path(figures_dir)
        figures_dir.mkdir(parents=True, exist_ok=True)

        order = list(summary["stream"]) if len(summary) else []
        if plot_clean:
            order += [name for name in sorted(paths) if name not in set(order)]
        if max_plots is not None:
            order = order[:max_plots]

        for name in order:
            of_stream = (intervals[intervals["stream"] == name] if len(intervals)
                         else intervals)
            figure = plot_stream(paths[name], name, of_stream, max_channels=max_channels)
            destination = figures_dir / "{0}.png".format(str(name).replace("/", "_"))
            figure.savefig(destination, dpi=120)
            plt.close(figure)
            written.append(destination)

        if show_progress:
            print("\nwrote {0} figure(s) to {1}".format(len(written), figures_dir))

    return intervals, summary


def main(argv=None):
    """Command line entry point."""
    parser = argparse.ArgumentParser(
        description="Report scored intervals and draw the streams they came from.")
    parser.add_argument("streams", nargs="+",
                        help="the stream files that were scored, or wildcard patterns")
    parser.add_argument("--scores", required=True, help="output of the scoring stage")
    parser.add_argument("--output", required=True,
                        help="per-interval report; .parquet or .csv by suffix")
    parser.add_argument("--summary", default=None, help="optional per-stream summary")
    parser.add_argument("--figures", default=None, help="directory to write figures to")
    parser.add_argument("--plot-clean", action="store_true",
                        help="also draw streams with nothing flagged")
    parser.add_argument("--max-plots", type=int, default=25,
                        help="cap on figures, most severe first (default 25)")
    parser.add_argument("--max-channels", type=int, default=5,
                        help="value channels drawn per stream (default 5)")
    parser.add_argument("--time-col", default=None, help="name of the time column")
    parser.add_argument("--stream-col", default=None,
                        help="column identifying the stream each row belongs to")
    parser.add_argument("--quiet", action="store_true", help="suppress progress output")

    args = parser.parse_args(argv)
    evaluate_scores(args.scores, expand_paths(args.streams), output_path=args.output,
                    summary_path=args.summary, figures_dir=args.figures,
                    plot_clean=args.plot_clean, max_plots=args.max_plots,
                    max_channels=args.max_channels, time_col=args.time_col,
                    stream_col=args.stream_col, show_progress=not args.quiet)


if __name__ == "__main__":
    main()
