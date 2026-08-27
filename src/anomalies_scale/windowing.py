"""Cut long streams into fixed-length windows, and fold the results back together afterwards.

Why this stage exists
---------------------
The corpus is built from every interval whose endpoints lie on a stream's dyadic grid, and that
grid is *relative* - `granularity` says how many cells, not how many points. So a corpus built
from streams of wildly differing lengths holds intervals of wildly differing lengths inside a
single depth band, and the nearest-neighbour search compares a candidate against whatever
happens to share its band.

Measured on C-MAPSS, where engine records run from 31 to 303 cycles, that is a ten-fold spread
*within* one depth. A 31-cycle degradation fills the whole of a short engine's depth-0 interval
and under a third of a long one's, so the same physical event produces two quite different
signatures and only one of them looks unusual. The pipeline's deficit against the notebook - the
notebook fixes its window at 33 points - was entirely recall, which is the shape of failure this
predicts.

Cutting every stream to one length collapses that spread. Depth *d* then means a fixed number of
points everywhere in the corpus, and the band a query lands in holds intervals of genuinely
comparable width.

The geometry
------------
A window carries ``size`` **increments**, hence ``size + 1`` points, and consecutive windows
**share their boundary point**::

    size = 16, a stream of 40 points (indices 0..39)

    w1   0 .......... 16      17 points
    w2        16 ..... 32     17 points     (point 16 belongs to both)
    w3             32..39      8 points     the remainder, kept as-is

Sharing the boundary is what makes the dyadic grid divide exactly. `grid_edges` places
``2**granularity + 1`` edges over a window, so 16 increments at granularity 3 give edges at
0, 2, 4, ..., 16 - eight cells of two increments each, no rounding. Take 16 *points* instead and
`np.linspace(0, 15, 9).round()` yields ``0 2 4 6 7 9 11 13 15``: one cell a single increment
wide, whose signature is systematically smaller than its band-mates for no reason connected to
the data. It also matches the convention used everywhere else in this project, where an interval
is inclusive at both ends so that adjacent intervals compose without a discontinuity.

The remainder is kept at whatever length is left over rather than padded. Padding by carrying
the last observation forward would append a run of zero increments, which signs as very nearly
nothing - and on run-to-failure data that run sits exactly where the failure is, so the padding
would manufacture normality precisely where the detector is supposed to fire.

A remainder still has to be long enough to carry the grid, which needs ``2**granularity + 1``
points. That is what `min_points` is for: the corpus side passes it and short tails are dropped,
while the scoring side leaves it at 2, because the widest-first search needs no grid and
discarding those points would mean never scoring the end of the stream.

Identity
--------
Each window becomes a stream in its own right, named ``{source}_{n}`` with *n* counting from 1 -
so windows of engine ``train_FD001__3`` are ``train_FD001__3_1``, ``..._2`` and so on. Three
further columns ride alongside: `source`, `window` and `offset`. They are not part of the
canonical format and every reader ignores columns it does not recognise, but they are what makes
the operation reversible, and what lets the calibration folds group on the engine rather than on
the window - sibling windows of one engine are not independent, and a fold that split them would
reintroduce the self-similarity the cross-validation exists to remove.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from anomalies_scale.canonical_streams import (
    CANONICAL_COLUMNS, STREAM_COLUMN, TIME_COLUMN, VALUES_COLUMN, as_matrix, read_canonical,
    write_canonical)

#: Value of `window.size` that switches the stage off, matching `umap.dimension`'s convention.
WINDOWING_OFF = -1

#: The stream a window was cut from - what folds and reassembly group on.
SOURCE_COLUMN = "source"
#: Which window of that stream this is, counting from 1.
WINDOW_COLUMN = "window"
#: Point index in the source stream that this window's point 0 corresponds to.
OFFSET_COLUMN = "offset"

#: The three columns added beside the canonical ones. Their presence is how a later stage
#: detects that it is looking at windows rather than whole streams.
WINDOW_COLUMNS = (SOURCE_COLUMN, WINDOW_COLUMN, OFFSET_COLUMN)


def resolve_size(value):
    """Read `window.size`: -1 for off, or a power of two.

    A power of two because the whole point is that the window divides cleanly - a size of 20
    with granularity 3 gives cells of 3, 2, 3, 2, 3, 2, 3, 2 increments and puts the ragged
    grid straight back. Rejecting it here is better than discovering the unevenness later as a
    band whose members are not the width the band claims.
    """
    size = int(value)
    if size == WINDOWING_OFF:
        return None
    if size < 2 or size & (size - 1):
        raise ValueError(
            "window.size is {0}; expected -1 to switch windowing off, or a power of two "
            "(16, 32, 64, ...). A size that is not a power of two cannot carry a dyadic grid "
            "evenly.".format(size))
    return size


def window_bounds(length, size):
    """Inclusive ``(lo, hi)`` point indices of every window over a stream of `length` points.

    Windows share their boundary point, so ``lo`` advances by exactly `size` and the last window
    runs to the end of the stream at whatever length remains. A stream of fewer than two points
    yields nothing - there is no path to sign.
    """
    last = int(length) - 1
    if last < 1:
        return []

    bounds, lo = [], 0
    while lo < last:
        hi = min(lo + int(size), last)
        bounds.append((lo, hi))
        lo = hi
    return bounds


def window_streams(canonical, size, min_points=2, output_path=None, show_progress=False):
    """Cut every stream of a canonical set into windows, and return them as a canonical set.

    Parameters
    ----------
    canonical : path or pd.DataFrame
        A canonical stream set - ``stream``, ``Time``, ``Stream``.
    size : int
        Increments per window; each full window therefore holds ``size + 1`` points.
    min_points : int
        Windows shorter than this are dropped. Only ever the final one, since every other
        window is full by construction. Pass ``2 ** granularity + 1`` on the corpus side, where
        a window too short for the dyadic grid cannot be signed at all, and leave it at 2 on
        the scoring side, where dropping it would mean never looking at the end of the stream.

    Returns
    -------
    pd.DataFrame
        Canonical columns plus :data:`WINDOW_COLUMNS`. `Time` is carried through unchanged -
        each window keeps the source stream's own clock rather than restarting at zero.
    """
    frame = read_canonical(canonical)
    size = int(size)

    rows, dropped, sources = [], 0, 0
    for source in frame.itertuples(index=False):
        name = str(getattr(source, STREAM_COLUMN))
        time = np.asarray(getattr(source, TIME_COLUMN), dtype=float)
        values = as_matrix(getattr(source, VALUES_COLUMN))
        sources += 1

        for number, (lo, hi) in enumerate(window_bounds(len(values), size), start=1):
            if hi - lo + 1 < int(min_points):
                dropped += 1
                continue
            rows.append({
                STREAM_COLUMN: "{0}_{1}".format(name, number),
                TIME_COLUMN: time[lo:hi + 1],
                VALUES_COLUMN: values[lo:hi + 1],
                SOURCE_COLUMN: name,
                WINDOW_COLUMN: number,
                OFFSET_COLUMN: int(lo),
            })

    if not rows:
        raise ValueError(
            "windowing at size {0} produced no windows from {1} stream(s). Every window was "
            "shorter than the {2}-point minimum - lower `signature.granularity`, or raise "
            "`window.size`.".format(size, sources, min_points))

    windows = pd.DataFrame(rows, columns=list(CANONICAL_COLUMNS) + list(WINDOW_COLUMNS))

    if show_progress:
        lengths = np.array([len(t) for t in windows[TIME_COLUMN]])
        full = int((lengths == size + 1).sum())
        print("{0} stream(s) -> {1} window(s) of {2} increments ({3} points); {4} full, "
              "{5} short".format(sources, len(windows), size, size + 1, full, len(windows) - full))
        if dropped:
            print("  dropped {0} tail window(s) under {1} points - too short to carry the "
                  "dyadic grid".format(dropped, min_points))
        print("  windows per stream {0}-{1}; points {2}-{3}".format(
            int(windows.groupby(SOURCE_COLUMN).size().min()),
            int(windows.groupby(SOURCE_COLUMN).size().max()),
            int(lengths.min()), int(lengths.max())))

    if output_path is not None:
        write_canonical(windows, output_path)

    return windows


def is_windowed(frame):
    """Whether a frame carries the window bookkeeping, and so can be reassembled."""
    return all(column in frame.columns for column in WINDOW_COLUMNS)


def merge_intervals(intervals):
    """Union inclusive ``[lo, hi]`` pairs, merging any that overlap or abut.

    Abutting counts as overlapping. Windows share their boundary point, so an anomaly running
    across a window edge is reported as ``[.., 16]`` by one window and ``[16, ..]`` by the next;
    left separate those would read as two findings rather than one, and the segment counts the
    evaluation reports would be wrong.
    """
    ordered = sorted(([int(lo), int(hi)] for lo, hi in intervals), key=lambda pair: pair[0])

    merged = []
    for lo, hi in ordered:
        if merged and lo <= merged[-1][1] + 1:
            merged[-1][1] = max(merged[-1][1], hi)
        else:
            merged.append([lo, hi])
    return merged


def reassemble(scored, anomalous_column="anomalous", score_column=None):
    """Fold scored windows back into their source streams.

    The inverse of :func:`window_streams`, applied to the frame the scorer produced. Each
    window's points are put back where they came from - dropping the boundary point every window
    after the first shares with its predecessor - and its intervals are shifted by its `offset`
    into source coordinates and unioned.

    This has to happen before anything is measured or drawn. Ground truth is stated per source
    stream, a window is not a thing anyone labelled, and metrics computed per window would count
    each shared boundary point twice.

    Returns the frame unchanged if it carries no window bookkeeping, so callers can apply it
    unconditionally.
    """
    frame = scored if isinstance(scored, pd.DataFrame) else pd.read_parquet(Path(scored))
    if not is_windowed(frame):
        return frame

    has_intervals = anomalous_column in frame.columns
    #: A per-point score travels the same way the values do - one entry per point, so it takes
    #: the same boundary-point drop. Without that it would be one longer than the stream it
    #: describes for every window after the first, and silently misalign against the labels.
    has_scores = bool(score_column) and score_column in frame.columns

    rows = []
    for name, group in frame.groupby(SOURCE_COLUMN, sort=True):
        group = group.sort_values(WINDOW_COLUMN)

        times, blocks, intervals, scores = [], [], [], []
        for position, window in enumerate(group.itertuples(index=False)):
            offset = int(getattr(window, OFFSET_COLUMN))
            time = np.asarray(getattr(window, TIME_COLUMN), dtype=float)
            values = as_matrix(getattr(window, VALUES_COLUMN))

            # Every window after the first opens on the point its predecessor closed on.
            start = 1 if position else 0
            times.append(time[start:])
            blocks.append(values[start:])

            if has_scores:
                scores.append(
                    np.asarray(getattr(window, score_column), dtype=float)[start:])

            if has_intervals:
                # Not `found or []`: parquet hands intervals back as a numpy array, and an
                # empty one raises rather than being falsy. Iterating it is fine either way.
                found = getattr(window, anomalous_column)
                for lo, hi in ([] if found is None else found):
                    intervals.append([int(lo) + offset, int(hi) + offset])

        row = {
            STREAM_COLUMN: str(name),
            TIME_COLUMN: np.concatenate(times),
            VALUES_COLUMN: np.vstack(blocks),
        }
        if has_intervals:
            row[anomalous_column] = merge_intervals(intervals)
        if has_scores:
            row[score_column] = np.concatenate(scores)
        rows.append(row)

    out = pd.DataFrame(rows)

    if has_intervals:
        flagged = [int(sum(hi - lo + 1 for lo, hi in v)) for v in out[anomalous_column]]
        out["n_anomalous_intervals"] = [len(v) for v in out[anomalous_column]]
        out["n_anomalous_points"] = flagged
        out["anomalous_fraction"] = [
            count / len(values) for count, values in zip(flagged, out[VALUES_COLUMN])]
        if "max_score" in frame.columns:
            out["max_score"] = [float(group["max_score"].max())
                                for _, group in frame.groupby(SOURCE_COLUMN, sort=True)]
    return out
