"""Derive an anomaly threshold per interval width from known-normal calibration streams.

A threshold has to answer "how far from the corpus does normal data sit?", and the only
honest way to measure that is on normal data the metric has not already been fitted to.
Measuring it on the corpus's own internal distances instead gives an answer that is
optimistic by an order of magnitude - the corpus is close to itself by construction - and a
detector calibrated that way flags normal data as a matter of course.

One threshold per interval width
--------------------------------
Thresholds are produced per dyadic depth rather than as a single scalar, because a level-*k*
signature term scales like (path increment) ** k and wide intervals therefore produce far
larger signature vectors than narrow ones. One global whitening does not remove that spread -
it applies the same linear map at every depth - so a single scalar threshold behaves partly
as a width filter rather than an anomaly test: a distance that is unremarkable for a
whole-stream interval can be wildly anomalous for a short one.

Where the calibration streams come from
---------------------------------------
Either given explicitly as a corpus of their own, or held out of the index. When held out,
the sample is stratified across interval widths: the shallow depths hold only a handful of
intervals per stream, and a uniform sample would thin them out of existence long before the
deep ones felt it, leaving exactly the widths that most need a threshold without one.

Calibration rows drawn from the index are scored with their own self-match skipped. A vector
searched against an index containing it matches itself at distance zero, so without that the
threshold would be calibrated on a column of zeros - a failure that looks like a working
pipeline right up until it flags everything.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from anomalies_scale.covariance_creation import load_covariance, read_corpus, signature_matrix
from anomalies_scale.index_creation import load_index, stratified_subsample, whiten


def parse_statistic(name):
    """Turn a statistic name into the function that reduces distances to a threshold.

    Accepts ``'max'``, ``'median'``, and any percentile written ``'p99'`` or ``'p99.9'``.

    ``max`` is the most permissive choice - it flags only intervals stranger than anything in
    the calibration set - and also the least stable, since it is decided entirely by the
    single worst calibration interval and inherits all of that interval's noise. A high
    percentile trades a little permissiveness for an estimate that does not move when one
    sample does.
    """
    if name == "max":
        return np.max
    if name == "median":
        return np.median
    if isinstance(name, str) and name.startswith("p"):
        try:
            percentile = float(name[1:])
        except ValueError:
            raise ValueError("could not read {0!r} as a percentile".format(name)) from None
        if not 0 <= percentile <= 100:
            raise ValueError("percentile must lie in [0, 100], got {0}".format(percentile))
        return lambda values: np.percentile(values, percentile)
    raise ValueError(
        "statistic must be 'max', 'median' or a percentile such as 'p99.9', got "
        "{0!r}".format(name))


def select_calibration_rows(index, calibration_split="withheld", holdout_fraction=0.2,
                            random_state=0):
    """Choose which rows of the index serve as calibration queries.

    Prefers rows the pipeline already held out - the `withheld` split exists for exactly this
    - and falls back to a stratified sample when the index carries no such split, so this
    still works on an index built from one corpus.

    Returns
    -------
    (np.ndarray, str)
        Row indices, and how they were chosen.
    """
    if calibration_split is not None:
        rows = np.flatnonzero(index.split == calibration_split)
        if rows.size:
            return rows, "split:{0}".format(calibration_split)

    # No dedicated split, so hold out a fraction of the index itself, stratified by depth.
    cap = int(round(holdout_fraction * index.depth.size))
    if cap < 1:
        raise ValueError(
            "holdout_fraction={0} leaves no calibration intervals out of {1}".format(
                holdout_fraction, index.depth.size))
    rows = stratified_subsample(index.depth, cap, random_state=random_state)
    return rows, "stratified holdout:{0:.0%}".format(holdout_fraction)


def calibration_distances(index, rows=None, queries=None, exclude_self=True):
    """Nearest-neighbour distance for each calibration interval, grouped by depth.

    Returns
    -------
    dict
        ``{depth: distances}``.
    """
    if (rows is None) == (queries is None):
        raise ValueError("give exactly one of rows (from the index) or queries (external)")

    if rows is not None:
        depths = index.depth[rows]
        vectors = index.vectors(rows)
    else:
        depths, vectors = queries

    distances = {}
    for depth in np.unique(depths):
        at_depth = np.flatnonzero(depths == depth)
        found, _ = index.search(vectors[at_depth], depth=int(depth), k=1,
                                exclude_self=exclude_self)
        distances[int(depth)] = found[:, 0]
    return distances


def calibrate_thresholds(index_path, meta_path, calibration=None, covariance=None,
                         statistic="max", calibration_split="withheld",
                         holdout_fraction=0.2, random_state=0, output_path=None,
                         show_progress=False):
    """Produce one threshold per interval width.

    Parameters
    ----------
    index_path, meta_path : str or Path
        The FAISS index and its row metadata, from :mod:`index_creation`.
    calibration : str or Path or pd.DataFrame, optional
        A corpus of known-normal calibration streams. When given, `covariance` is required,
        since these signatures must be whitened into the same space as the index before they
        can be compared with it. When omitted, calibration rows are taken out of the index.
    covariance : str or Path or np.ndarray, optional
        ``Sigma^-1/2``, needed only to whiten an external calibration corpus.
    statistic : str
        How distances at one width become that width's threshold. See
        :func:`parse_statistic`.

    Returns
    -------
    pd.DataFrame
        One row per depth: the interval width it corresponds to, how many calibration
        intervals it was measured on, the threshold, and the distance distribution behind it.
    """
    index = load_index(index_path, meta_path)
    reduce_to_threshold = parse_statistic(statistic)

    if calibration is not None:
        if covariance is None:
            raise ValueError(
                "an external calibration corpus needs the covariance matrix too: its "
                "signatures have to be whitened into the same space as the index before "
                "any distance between them means anything")

        frame = read_corpus(calibration)
        signatures, columns = signature_matrix(frame)
        if isinstance(covariance, (str, Path)):
            covariance = load_covariance(covariance, expected_columns=columns)
        vectors = whiten(signatures, covariance).astype("float32")
        distances = calibration_distances(
            index, queries=(frame["depth"].to_numpy(dtype=int), vectors),
            exclude_self=False)
        source = "corpus:{0}".format(
            Path(calibration).name if isinstance(calibration, (str, Path)) else "frame")
        widths = frame.groupby("depth")["n_points"].median() if "n_points" in frame else None
    else:
        rows, source = select_calibration_rows(
            index, calibration_split, holdout_fraction, random_state)
        distances = calibration_distances(index, rows=rows, exclude_self=True)
        widths = None

    if widths is None:
        # Recover a representative width per depth from the index's own interval endpoints.
        spans = pd.DataFrame({"depth": index.depth,
                              "points": np.asarray(index.identity.get("hi", index.depth))
                                        - np.asarray(index.identity.get("lo", index.depth)) + 1})
        widths = spans.groupby("depth")["points"].median()

    records = []
    for depth in sorted(distances):
        values = np.asarray(distances[depth], dtype=float)
        records.append({
            "depth": depth,
            "relative_width": 2.0 ** -depth,
            "median_points": float(widths.get(depth, np.nan)),
            "n_calibration": int(values.size),
            "threshold": float(reduce_to_threshold(values)),
            "median_distance": float(np.median(values)),
            "p99_distance": float(np.percentile(values, 99)),
            "max_distance": float(values.max()),
        })

    table = pd.DataFrame.from_records(records)
    table.attrs["statistic"] = statistic
    table.attrs["source"] = source

    if show_progress:
        print("calibrated on {0} ({1} intervals) using statistic {2!r}".format(
            source, int(table["n_calibration"].sum()), statistic))
        print(table.to_string(index=False))
        spread = table["threshold"].max() / max(table["threshold"].min(), 1e-30)
        print("\nthresholds span {0:.0f}x across widths, which is why they are per depth "
              "rather than one scalar".format(spread))

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(output_path, index=False)
        if show_progress:
            print("wrote {0}".format(output_path))

    return table


def load_thresholds(path):
    """Read a threshold table back as ``{depth: threshold}``."""
    table = pd.read_csv(path)
    return dict(zip(table["depth"].astype(int), table["threshold"].astype(float)))


def main(argv=None):
    """Command line entry point."""
    parser = argparse.ArgumentParser(
        description="Calibrate one anomaly threshold per interval width.")
    parser.add_argument("--index", required=True, help="FAISS index from index_creation")
    parser.add_argument("--index-meta", required=True, help="its npz of row metadata")
    parser.add_argument("--calibration", default=None,
                        help="corpus of known-normal calibration streams; omit to hold rows "
                             "out of the index instead")
    parser.add_argument("--covariance", default=None,
                        help="Sigma^-1/2 CSV, required with --calibration")
    parser.add_argument("--statistic", default="max",
                        help="max, median, or a percentile such as p99.9 (default max)")
    parser.add_argument("--calibration-split", default="withheld",
                        help="index split to calibrate on; falls back to a stratified "
                             "holdout when the index has no such split")
    parser.add_argument("--holdout-fraction", type=float, default=0.2,
                        help="fraction held out when there is no calibration split")
    parser.add_argument("--random-state", type=int, default=0, help="seed for that holdout")
    parser.add_argument("--output", required=True, help="CSV of thresholds to write")
    parser.add_argument("--quiet", action="store_true", help="suppress progress output")

    args = parser.parse_args(argv)
    calibrate_thresholds(
        args.index, args.index_meta, calibration=args.calibration,
        covariance=args.covariance, statistic=args.statistic,
        calibration_split=args.calibration_split, holdout_fraction=args.holdout_fraction,
        random_state=args.random_state, output_path=args.output,
        show_progress=not args.quiet)


if __name__ == "__main__":
    main()
