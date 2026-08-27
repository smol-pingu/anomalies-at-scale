"""Build a pooled signature corpus from a canonical stream set.

Takes the canonical ``(stream, Time, Stream)`` table, cuts every stream into its intervals,
signs each one with a base point, and pools them into a single table whose layout
:func:`~anomalies_scale.anomaly_detection_pooled.split_corpus` and
:class:`~anomalies_scale.anomaly_detection_pooled.PooledCorpus` already understand.

Which intervals
---------------
Every interval whose two endpoints lie on the finest dyadic grid - not only the dyadic tree.
The tree is what a recursive bisection produces and what most signature libraries offer, but
it is not what this project searches. The widest-first search extends a clean block outward and
asks about runs of adjacent cells, and a run of three cells is not a dyadic node; scoring it
against a corpus holding only 2-cell and 4-cell intervals compares it to the wrong width. At
granularity 3 that is 36 intervals per stream against the tree's 15, and the tree is included -
a dyadic node is a run of aligned cells.

How they are signed
-------------------
Directly, one ``iisignature.sig`` call per interval, over its own raw points. Nothing is
composed from anything else.

The alternative would be Chen's identity: sign the ``2**granularity`` leaves once and build
every wider run by combining. It costs about one pass over each stream instead of fifteen, and
it is exact in exact arithmetic - but not in floating point. ``iisignature.sigcombine`` returns
float64 and is accurate to roughly float32, measured on this data at 2.45e-08 relative, and
that is a *floor* rather than an accumulation: one combine costs as much precision as seven.
Composition therefore quietly caps how deep a truncation is worth taking, because the terms a
deeper truncation adds are the small ones that noise reaches first. Signing directly costs
arithmetic and buys back the digits.

Time and base points
--------------------
The canonical set carries time in its own column. When it is present and finite it is prepended
as channel 0, so the signature integrates against the real, possibly irregular timebase. When
it is absent a normalised index stands in, so that every stream still has a time channel and
the widths agree.

The base point is applied by prepending the origin to each interval before signing it, so the
path signed is ``0 -> x_lo -> ... -> x_hi``. Anchoring absolute position this way is what stops
a signature being translation invariant and therefore blind to a level shift. Doing it inside
the single ``sig`` call, rather than by combining a jump onto a finished signature, means it
enters exactly once by construction - there is no operation left that could apply it twice.
"""

from __future__ import annotations

from pathlib import Path

import iisignature
import numpy as np
import pandas as pd

from anomalies_scale.canonical_streams import (
    STREAM_COLUMN, TIME_COLUMN, VALUES_COLUMN, as_matrix, read_canonical)

#: Metadata carried alongside the signature terms. ``depth`` is the one the search reads.
META_COLUMNS = ("depth", "stream", "lo", "hi", "n_cells")


def grid_edges(length, granularity):
    """Point indices of the finest dyadic grid over a stream of `length` points."""
    cells = 2 ** granularity
    if length < cells + 1:
        raise ValueError(
            "a stream of {0} point(s) cannot carry {1} dyadic cells; lower `granularity` or "
            "use longer streams".format(length, cells))
    return np.linspace(0, length - 1, cells + 1).round().astype(int)


def grid_intervals(length, granularity):
    """Every interval with both endpoints on the dyadic grid, as ``(depth, lo, hi, n_cells)``.

    Depth is assigned by the same nearest-log2 rule ``PooledCorpus.depth_for_width`` uses at
    query time, so a run of *k* cells out of ``2**g`` lands at ``round(g - log2 k)``. Corpus
    and query agree by construction, and the bands that previously held only powers of two now
    hold genuine intervals of the intermediate widths.

    ``lo`` and ``hi`` are inclusive point indices, so adjacent intervals share a boundary point
    and composing them introduces no discontinuity.
    """
    edges = grid_edges(length, granularity)
    cells = 2 ** granularity

    intervals = []
    for first in range(cells):
        for last in range(first, cells):
            run = last - first + 1
            intervals.append((int(round(granularity - np.log2(run))),
                              int(edges[first]), int(edges[last + 1]), run))
    intervals.sort(key=lambda record: (record[0], record[1]))
    return intervals


def base_pointed_signature(piece, trunc):
    """Signature of one interval, base-pointed, in a single call.

    The base point is applied by prepending the origin to the interval's own points, so what
    gets signed is the path ``0 -> x_lo -> ... -> x_hi``. That is the whole of it: one
    ``iisignature.sig`` call, no composition anywhere, and the base point enters exactly once
    by construction rather than by an operation that could be repeated.
    """
    piece = np.asarray(piece, dtype=float)
    return iisignature.sig(np.vstack([np.zeros((1, piece.shape[1])), piece]), trunc)


def interval_signatures(path, trunc, granularity):
    """Base-pointed signature of every grid interval of one stream.

    Every interval is signed from its own raw points. Returns
    ``(depth, lo, hi, n_cells, signature)`` per interval, sorted by depth then start.
    """
    path = np.asarray(path, dtype=float)
    if path.ndim != 2:
        raise ValueError("path must be 2-D (length, width), got {0}".format(path.shape))

    return [(depth, lo, hi, run, base_pointed_signature(path[lo:hi + 1], trunc))
            for depth, lo, hi, run in grid_intervals(len(path), granularity)]


def stream_paths(canonical):
    """Yield ``(name, path)`` from a canonical set, with a time channel in column 0.

    A time column that is missing, the wrong length, or entirely non-finite is treated as
    absent rather than as an error - some sources genuinely have no clock - and a normalised
    index stands in so that every stream still has a time channel of the same width.
    """
    frame = read_canonical(canonical)
    for row in frame.itertuples(index=False):
        values = as_matrix(getattr(row, VALUES_COLUMN))
        time = np.asarray(getattr(row, TIME_COLUMN), dtype=float)

        usable = time.size == len(values) and np.isfinite(time).all()
        if not usable:
            time = np.linspace(0.0, 1.0, len(values))
        yield str(getattr(row, STREAM_COLUMN)), np.hstack([time.reshape(-1, 1), values]), usable


def compute_corpus(canonical, trunc, granularity, output_path=None, show_progress=False):
    """Build the pooled signature corpus from a canonical stream set.

    Returns
    -------
    pd.DataFrame
        One row per interval: `depth`, `stream`, `lo`, `hi`, `n_cells`, then `sig_0..sig_n`.
    """
    streams = list(stream_paths(canonical))
    if not streams:
        raise ValueError("the canonical set holds no streams")

    widths = {path.shape[1] for _, path, _ in streams}
    if len(widths) > 1:
        raise ValueError(
            "streams disagree about channel count ({0}); they cannot share one corpus".format(
                sorted(widths)))
    width = widths.pop()
    n_terms = iisignature.siglength(width, trunc)

    untimed = sum(1 for _, _, usable in streams if not usable)
    if show_progress:
        print("{0} stream(s), width {1}, truncation {2} -> {3} term(s), granularity {4}, "
              "{5}".format(len(streams), width, trunc, n_terms, granularity,
                           "signed directly"))
        if untimed:
            print("  {0} stream(s) had no usable time column; a normalised index stands "
                  "in".format(untimed))

    rows, meta = [], []
    for number, (name, path, _) in enumerate(streams, start=1):
        for depth, lo, hi, run, signature in interval_signatures(
                path, trunc, granularity):
            rows.append(signature)
            meta.append((depth, name, lo, hi, run))
        if show_progress and number % 25 == 0:
            print("  signed {0}/{1} stream(s)".format(number, len(streams)))

    corpus = pd.DataFrame(np.asarray(rows, dtype=float),
                          columns=["sig_{0}".format(i) for i in range(n_terms)])
    for position, name in enumerate(META_COLUMNS):
        corpus.insert(position, name, [record[position] for record in meta])

    if show_progress:
        print("corpus {0:,} interval(s) x {1} term(s)".format(len(corpus), n_terms))
        print(corpus["depth"].value_counts().sort_index().to_string())

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        corpus.to_parquet(output_path, index=False)

    return corpus


# ---------------------------------------------------------------------------------------
# Carried over from the old `create_corpus_merged`
#
# `add_base_point` attaches a base point by Chen-combining a jump from the origin onto a
# finished signature. This module's own `base_pointed_signature` does the same thing by
# prepending the origin to the path and signing once, which is exact where this carries
# `sigcombine`'s float32-accuracy floor. Kept because the per-interval detector still uses it;
# prefer `base_pointed_signature` in anything new.
# ---------------------------------------------------------------------------------------

#: How a finished corpus is written, dispatched on the output suffix.
CORPUS_WRITERS = {
    '.parquet': lambda frame, path: frame.to_parquet(path, index=False),
    '.pq': lambda frame, path: frame.to_parquet(path, index=False),
    '.csv': lambda frame, path: frame.to_csv(path, index=False),
}


def write_corpus(frame, output_path):
    """Write a corpus as parquet or CSV, chosen by the output's suffix.

    Parquet is the better default by some margin - a corpus is a wide float table, and CSV
    both inflates it several-fold and round-trips through decimal text - but CSV is offered
    because it can be read without a parquet engine.
    """
    output_path = Path(output_path)
    writer = CORPUS_WRITERS.get(output_path.suffix.lower())
    if writer is None:
        raise ValueError(
            'unsupported corpus format {0!r} for {1}; expected one of {2}'.format(
                output_path.suffix, output_path.name, sorted(CORPUS_WRITERS)))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer(frame, output_path)
    return output_path


def add_base_point(signature, path_at_lo, width, trunc):
    """Chen-combine a jump from the origin onto an interval's signature.

    ``signature`` describes the interval's shape; combining it with the signature of the
    straight jump from the origin to ``path_at_lo`` (the interval's actual first point)
    anchors its absolute position, breaking the translation invariance described in the
    module docstring.

    Computed per interval rather than precomputed, since the jump depends on where the
    specific interval starts.
    """
    path_at_lo = np.asarray(path_at_lo, dtype=float)
    origin = np.zeros_like(path_at_lo)
    jump = iisignature.sig(np.stack([origin, path_at_lo], axis=-2), trunc)
    return iisignature.sigcombine(jump, signature, width, trunc)

