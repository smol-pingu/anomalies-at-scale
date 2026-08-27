"""Bring raw streams into a canonical shape, reject the ones that cannot be signed, and split.

Four responsibilities, all of which have to happen before anything else in the pipeline can
touch the data.

**Naming.** Benchmark data routinely arrives without a header - SMD's files are read with
``header=None``, so pandas labels the columns ``0..37`` - and positional integers are a poor
thing to carry through six stages. Columns are given canonical names: the time channel, if
there is one, becomes ``Time``, and the value channels become ``Variable_1``, ``Variable_2``
and so on in file order. Frames that already have real names are left exactly as they are;
inventing names over the top of somebody's own would be worse than useless.

**Screening.** A signature is an integral along a path, so a single missing point does not
degrade it - it destroys it. One NaN propagates through ``iisignature.sig`` into every term
of that interval, through Chen's identity into every interval containing it, and from there
into the covariance, where it takes the whole metric with it. There is no partial recovery
from that, and no honest way to guess the missing value, so any stream carrying a non-finite
value is removed and reported by name rather than being quietly patched.

Removals are returned, not just printed. A run that silently discarded half its corpus should
not look the same afterwards as one that discarded nothing.

**Lagging.** Optional, off by default. Every value channel gains a copy of itself running behind
it, widening the path from ``1 + k`` to ``1 + 2k``. A signature integrates each channel against
every other but never against itself - ``S(i,i)`` is half the square of the total increment,
decided by the endpoints alone - so how a channel got where it went is invisible at level 2.
Pairing it with its own past makes ``S(i, i_lag)`` a real cross term that does depend on the
route.

Two constructions, chosen by `lag_method` (see :data:`LAG_METHODS`). :func:`delay_columns`
appends the delayed copy on the original grid, which is cheap and leaves every point index
meaning what it meant. :func:`lead_lag_columns` is the interleaved form SigNova uses: exact -
its signed area is precisely half the quadratic variation - at the cost of doubling the path,
after which point indices refer to lead-lag positions and have to be collapsed back before
anything is measured.

**Splitting.** Two sets come out: `corpus`, the data assumed normal, and `test`, what gets
judged. Usually that division has already been made - labelled benchmarks ship the two sides as
distinct periods - and this stage only has to make it when they have not, by cutting the test
period off the end of each corpus series.

There used to be a third set. `fit` fitted the metric and `withheld` calibrated the threshold
against normal data the metric had never seen, because deriving a threshold from the corpus's
own nearest-neighbour distances understates it badly - by roughly 13x on SMD, which put the
threshold below the median of normal-but-unseen intervals.
:mod:`anomalies_scale.crossvalidated_thresholding` now answers that question by k-fold over
streams: every interval takes a turn scored against an index built from the other folds, so
*every* stream is calibrated under the condition the detector actually meets, rather than only
the 20% that happened to fall in the holdout. A fixed fit/withheld cut was the weaker version of
the same idea and has been removed.

When the corpus/test cut does have to be made here it is **positional, never random**. These are
time series, and a random split would put two neighbouring stretches of one series on opposite
sides of the boundary, which leaks the answer across it and makes the test period look far more
familiar than genuinely unseen data would.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from anomalies_scale.canonical_streams import DEFAULT_TIME_COLUMNS, read_frame

#: What the time channel is called once a nameless stream has been named.
TIME_NAME = "Time"

#: How the value channels are named, numbered from 1 in file order.
VARIABLE_TEMPLATE = "Variable_{0}"


def is_positional(column):
    """Whether one column label is a position rather than a name.

    Integers are the obvious case - ``pd.read_csv(..., header=None)`` labels columns with a
    RangeIndex. Their *string* forms count too, because parquet has no integer column labels:
    a headerless file written to the project's default format comes back with ``'0'``,
    ``'1'``, ``'2'``, which is exactly as positional as it went in and would otherwise be
    mistaken for somebody's own naming.
    """
    if isinstance(column, (int, np.integer)):
        return True
    return str(column).lstrip("-").isdigit()


def has_column_names(frame):
    """Whether a frame arrived with real column names or just positional labels."""
    return not all(is_positional(column) for column in frame.columns)


def existing_time_name(frame, time_col=None):
    """The name of the time column in an already-named frame, or None if it has none."""
    if time_col is not None:
        if time_col not in frame.columns:
            raise KeyError(
                "time column {0!r} not found; available columns: {1}".format(
                    time_col, list(frame.columns)))
        return time_col

    lowered = {str(column).lower(): column for column in frame.columns}
    for candidate in DEFAULT_TIME_COLUMNS:
        if candidate in lowered:
            return lowered[candidate]
    return None


def name_columns(frame, time_col=None):
    """Give a stream canonical column names when it arrived without any.

    Parameters
    ----------
    frame : pd.DataFrame
        One stream's rows.
    time_col : str or int, optional
        Which column holds time. A name for a frame that has names, or a position for one
        that does not - a nameless file cannot be asked about by name.

    Returns
    -------
    (pd.DataFrame, str or None)
        The frame, and the name of its time column if it has one.
    """
    if has_column_names(frame):
        return frame, existing_time_name(frame, time_col)

    position = None
    if time_col is not None:
        if isinstance(time_col, (int, np.integer)) or str(time_col).lstrip("-").isdigit():
            position = int(time_col)
            if not -len(frame.columns) <= position < len(frame.columns):
                raise IndexError(
                    "time column position {0} is outside the {1} column(s) present".format(
                        position, len(frame.columns)))
            position %= len(frame.columns)
        else:
            raise KeyError(
                "this stream has no column names, so its time column cannot be identified "
                "by the name {0!r}. Give raw.time_column as a position instead, or leave it "
                "null if there is no time channel.".format(time_col))

    names, variable = [], 0
    for column in range(len(frame.columns)):
        if column == position:
            names.append(TIME_NAME)
        else:
            variable += 1
            names.append(VARIABLE_TEMPLATE.format(variable))

    frame = frame.copy()
    frame.columns = names
    return frame, TIME_NAME if position is not None else None


#: How a lagged channel is named, from the channel it was delayed from and the delay.
LAG_TEMPLATE = "{0}_lag{1}"

#: The two ways of pairing a channel with its own past.
#:
#:   ``delay``     append a copy shifted back `lag` steps, on the original time grid. Path
#:                 length is unchanged, so point indices keep meaning what they meant.
#:   ``lead_lag``  the interleaved construction SigNova uses: double the grid and advance the
#:                 lead and the lag alternately. Exact, and twice as long.
LAG_METHODS = ("delay", "lead_lag")


def lead_lag_columns(frame, name=""):
    """Interleave a channel with its own past on a doubled time grid.

    The construction from ``SigNova._path_transforms``::

        stream = np.repeat(stream, 2, axis=0)
        stream = np.column_stack((stream[1:, :], stream[:-1, :]))

    An *n*-point stream becomes ``2n - 1`` points, alternately advancing the lead and letting
    the lag catch up::

        j=0  (x0, x0)     j=1  (x1, x0)     j=2  (x1, x1)     j=3  (x2, x1)  ...

    Why bother, given :func:`delay_columns` already pairs a channel with its past. Because this
    version is *exact*: the signed area of the lead-lag path is precisely half the channel's
    quadratic variation, whereas the delay embedding on a shared grid only orders paths by it.
    On the 33-point test pair, sum of squared increments 1.788 against a delay-embedding area of
    0.179 - an order of magnitude out.

    What it costs, beyond twice the points. Every point index downstream - the dyadic grid,
    window boundaries, the ``[lo, hi]`` a flag is reported in - now refers to a lead-lag index
    rather than an observation. Observation *i* occupies indices ``{2i - 1, 2i}`` (and ``{0}``
    for the first), so the mapping is exact and invertible;
    :func:`anomalies_scale.stream_evaluation.collapse_lead_lag` is what inverts it, and it has
    to run before anything is measured against per-observation labels.
    """
    time_name = existing_time_name(frame)
    values = [column for column in frame.columns if column != time_name]
    if not len(frame):
        raise ValueError("stream {0!r} has no rows to transform".format(name or "?"))

    block = np.repeat(frame[values].to_numpy(dtype=float), 2, axis=0)
    widened = pd.DataFrame(
        np.column_stack([block[1:], block[:-1]]),
        columns=values + [LAG_TEMPLATE.format(column, 1) for column in values])

    if time_name is not None:
        # The lead's clock. On the steps where only the lag advances it does not move, which is
        # what the construction means: those steps happen at one instant.
        doubled = np.repeat(frame[time_name].to_numpy(dtype=float), 2)
        widened.insert(0, time_name, doubled[1:])

    return widened


def delay_columns(frame, lag, name=""):
    """Append a delayed copy of every value channel, widening the stream from k to 2k.

    The delay embedding. A signature already integrates each channel against every other, but a
    channel is never integrated against *itself* - the level-2 term ``S(i,i)`` is exactly half
    the square of the total increment, so it is decided by the endpoints alone and carries no
    ordering. Two windows that start and end at the same values therefore sign **identically**
    at truncation 2, however differently they got there. Pairing a channel with a copy of itself
    shifted back `lag` steps makes ``S(i, i_lag)`` a genuine cross term, whose antisymmetric
    part does depend on the route.

    This is the lead-lag idea of the rough-paths literature, but not its exact form, and the
    difference is worth being precise about. The textbook construction - :func:`lead_lag_columns`
    here, and what SigNova implements - interleaves lead and lag on a *doubled* time grid, and
    its signed area is then exactly half the channel's quadratic variation. Appending a column on
    the original grid is cheaper: path lengths, the dyadic grid, and the point indices a flag is
    reported in all stay as they were, so nothing downstream needs to know it happened. But it
    only approximates the quantity. Measured on a 33-point test path with the same endpoints as
    a straight line: ``sum of squared increments`` 1.788, signed area under this construction
    0.179. The ordering it induces is what earns its place, not the identity.

    What that buys, measured on the same pair of paths: without a lag the smooth and the jagged
    version are bit-identical at truncation 2; with one they separate about six-fold.

    The first `lag` rows have no history, and take the stream's **initial value** - the reading
    is treated as having been stationary at that value for the `lag` steps before the record
    begins. That is an assumption rather than an observation, and it shows up as zero increment
    in the lagged channels over the opening steps, but it is a far cheaper one than the
    alternative: dropping those rows discards real measurements, and shortens every stream,
    which puts the point indices out of step with any externally supplied labels.

    Note this widens the path from ``1 + k`` to ``1 + 2k``, and signature dimension is roughly
    ``width ** trunc`` - so the corpus grows by about ``2 ** trunc``. At truncation 2 on
    C-MAPSS's 24 channels that is 650 terms becoming 2,450.
    """
    if not lag:
        return frame

    lag = int(lag)
    if lag < 0:
        raise ValueError("lag must be non-negative, got {0}".format(lag))
    if not len(frame):
        raise ValueError("stream {0!r} has no rows to lag".format(name or "?"))

    time_name = existing_time_name(frame)
    values = [column for column in frame.columns if column != time_name]

    # Filled from the first row explicitly rather than by `bfill`, which has nothing to fill
    # from when `lag` is at least the length of the stream - every row is NaN after the shift,
    # and the whole stream should then read as its own initial value.
    lagged = frame[values].shift(lag).fillna(frame[values].iloc[0])
    lagged.columns = [LAG_TEMPLATE.format(column, lag) for column in values]

    return pd.concat([frame, lagged], axis=1)


def lag_streams(streams, lag, method="delay", show_progress=True):
    """Pair every channel with its own past, by whichever construction `method` names.

    See :data:`LAG_METHODS`. ``lead_lag`` is the exact one and doubles the path; ``delay`` is
    the cheap one and leaves point indices alone.
    """
    if not lag:
        return streams

    if method not in LAG_METHODS:
        raise ValueError(
            "preprocess.lag_method is {0!r}; expected one of {1}".format(
                method, list(LAG_METHODS)))

    if method == "lead_lag":
        if int(lag) != 1:
            raise ValueError(
                "the lead-lag construction interleaves a stream with itself one step behind, "
                "so it only has a lag of 1; preprocess.lag is {0}. Use lag_method 'delay' for "
                "longer delays.".format(lag))
        widened = [(name, lead_lag_columns(frame, name)) for name, frame in streams]
    else:
        widened = [(name, delay_columns(frame, lag, name)) for name, frame in streams]

    if show_progress and widened:
        before, after = streams[0][1], widened[0][1]
        time_name = existing_time_name(after)
        channels = sum(1 for c in after.columns if c != time_name)
        print("{0} (lag {1}): {2} value channel(s) -> {3}, path width {4} -> {5}".format(
            method, lag, channels // 2, channels, len(before.columns), len(after.columns)))
        if method == "lead_lag":
            print("  path length {0} -> {1} point(s); every index downstream is now a lead-lag "
                  "index, collapsed back before evaluation".format(len(before), len(after)))
        else:
            print("  stream lengths unchanged; the first {0} point(s) of each take the "
                  "stream's initial value as their history".format(lag))
    return widened


def nonfinite_report(frame):
    """Describe the non-finite values in a stream, or return None if there are none.

    Reports NaN and infinity separately. They usually mean different things - a gap in
    collection against a division that went wrong upstream - and neither can be signed.
    """
    numeric = frame.select_dtypes(include="number")
    if numeric.empty:
        return None

    values = numeric.to_numpy(dtype=float)
    nan = np.isnan(values)
    infinite = np.isinf(values)
    if not (nan.any() or infinite.any()):
        return None

    affected = numeric.columns[(nan | infinite).any(axis=0)]
    rows = np.flatnonzero((nan | infinite).any(axis=1))
    return {
        "n_nan": int(nan.sum()),
        "n_inf": int(infinite.sum()),
        "n_rows_affected": int(rows.size),
        "first_row": int(rows[0]),
        "columns": ", ".join(str(column) for column in affected),
        "reason": "NaN" if nan.any() and not infinite.any()
                  else ("infinity" if infinite.any() and not nan.any() else "NaN and infinity"),
    }


def screen_streams(streams, show_progress=True):
    """Split named streams into those that can be signed and those that cannot.

    Parameters
    ----------
    streams : iterable of (str, pd.DataFrame)
        Named streams, already through :func:`name_columns`.

    Returns
    -------
    (list, pd.DataFrame)
        The streams that survive, and a table of what was removed and why. The table has its
        columns even when nothing was removed, so a downstream reader never has to special
        case the happy path.
    """
    kept, removed = [], []

    for name, frame in streams:
        report = nonfinite_report(frame)
        if report is None:
            kept.append((name, frame))
            continue

        removed.append({"stream": name, "n_points": len(frame), **report})
        if show_progress:
            print("{0} removed due to {1} ({2} value(s) across {3} row(s), first at row "
                  "{4}, in column(s) {5})".format(
                      name, report["reason"], report["n_nan"] + report["n_inf"],
                      report["n_rows_affected"], report["first_row"], report["columns"]))

    table = pd.DataFrame(
        removed,
        columns=["stream", "n_points", "n_nan", "n_inf", "n_rows_affected", "first_row",
                 "columns", "reason"])

    if show_progress and len(removed):
        print("\n{0} stream(s) removed; {1} remain".format(len(removed), len(kept)))

    return kept, table


# ---------------------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------------------
def cut_in_time(name, frame, fraction):
    """Cut one stream at `fraction` of its length, leading part first.

    The one division this stage still makes: when a dataset ships no separate test period, the
    corpus series has to supply one, and it comes off the end. Both halves keep the stream's
    name with a suffix, so a file written from either can still be traced back to the series it
    came from.
    """
    cut = int(round(float(fraction) * len(frame)))
    cut = min(max(cut, 1), len(frame) - 1)
    return ((name + "_a", frame.iloc[:cut].reset_index(drop=True)),
            (name + "_b", frame.iloc[cut:].reset_index(drop=True)))


# ---------------------------------------------------------------------------------------
# Reading and writing
# ---------------------------------------------------------------------------------------
def resolve_files(source):
    """Accept a directory, a single path, or a list of paths, and return the files.

    The `umap` stage writes a directory of per-stream parquet while the raw side is a list of
    downloaded files, so this stage has to take either without caring which.
    """
    if isinstance(source, (str, Path)):
        path = Path(source)
        return sorted(p for p in path.iterdir() if p.is_file()) if path.is_dir() else [path]
    return [Path(item) for item in source]


def load_named_streams(files, time_col=None, value_cols=None, stream_col=None):
    """Read every stream from every file, canonically named.

    A file may hold one stream or many; `stream_col` names the column that distinguishes them,
    and is dropped once it has been used, since it is an identifier rather than a channel and
    signing it would be meaningless.

    Stream names are qualified by the file stem, because stream identifiers only have to be
    unique within their own file while the sets built here pool every file together.
    """
    from anomalies_scale.canonical_streams import is_canonical, iter_frames

    streams = []
    for path in resolve_files(files):
        # A canonical set already carries its streams one per row, named, with time resolved.
        # It needs no naming pass and no stream column - that work was done when it was built.
        if is_canonical(path):
            for name, block in iter_frames(path):
                streams.append((name, block))
            continue

        frame = read_frame(path)
        stem = Path(path).stem

        groups = ([(stem, frame)] if stream_col is None or stream_col not in frame.columns
                  else [("{0}__{1}".format(stem, key), group.drop(columns=[stream_col]))
                        for key, group in frame.groupby(stream_col, sort=True)])

        for name, group in groups:
            named, time_name = name_columns(group.reset_index(drop=True), time_col)
            if value_cols:
                keep = ([time_name] if time_name else []) + [
                    c for c in named.columns if c in set(value_cols) and c != time_name]
                missing = set(value_cols) - set(named.columns)
                if missing:
                    raise KeyError(
                        "raw.value_column names {0}, which {1} does not have; its columns are "
                        "{2}".format(sorted(missing), name, list(named.columns)))
                named = named[keep]
            streams.append((name, named))
    return streams


def write_canonical_set(streams, path):
    """Write named streams as one canonical set.

    The split artifacts are canonical files rather than directories of per-stream parquet,
    because that is what `signature_computer` reads. One file per split also means a split is
    a single Snakemake output rather than a directory whose staleness has to be reasoned
    about - a rerun that produces fewer streams simply writes a shorter table.
    """
    from anomalies_scale.canonical_streams import to_canonical, write_canonical

    return write_canonical(to_canonical(streams, time_col=TIME_NAME), path), len(streams)


def write_streams(streams, directory):
    """Write one parquet per stream, clearing anything stale first.

    Snakemake declares this rule's outputs as directories, so a rerun that produces fewer
    streams than the last one would otherwise leave the surplus behind and every later stage
    would silently read them.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for stale in directory.glob("*.parquet"):
        stale.unlink()

    for name, frame in streams:
        frame.to_parquet(directory / "{0}.parquet".format(name), index=False)
    return len(streams)


# ---------------------------------------------------------------------------------------
# Stage entry point
# ---------------------------------------------------------------------------------------
def corpus_and_test(corpus_files, test_files, base_fraction=0.572, time_col=None,
                    value_cols=None, stream_col=None, lag=0, lag_method='delay',
                    show_progress=True):
    """Name, screen, and make the corpus/test division. Shared by both entry points.

    Two shapes of input, and the difference is whether that division has already been made:

    **Test files supplied.** The two sides are already distinct periods - which is how labelled
    benchmarks ship - so `base_fraction` is not used at all.

    **No test files.** The test period is taken off the end of each corpus series at
    `base_fraction`. This is the shape where positional cutting matters most, since corpus and
    test are then consecutive stretches of one series.

    Screening happens before any division, deliberately. A stream carrying a non-finite value
    is removed outright, and removing it afterwards would leave the fractions describing a
    corpus that no longer exists.

    Lagging happens after screening and before the division, for two reasons. `shift` would
    spread an existing NaN into a second row, so screening has to have run first; and cutting a
    single series into corpus and test *after* lagging means the test half's first lagged values
    come from the last rows of the corpus half, which is not a leak - they are consecutive
    points of one series, and a detector running online would have that history.
    """
    corpus = load_named_streams(corpus_files, time_col, value_cols, stream_col)
    if show_progress:
        print("read {0} corpus stream(s)".format(len(corpus)))

    corpus, corpus_removed = screen_streams(corpus, show_progress=show_progress)
    if not corpus:
        raise ValueError(
            "every corpus stream was removed as unsignable; there is nothing to build a "
            "corpus of normality from. See the removal table for what went wrong.")

    corpus = lag_streams(corpus, lag, lag_method, show_progress=show_progress)

    supplied = list(resolve_files(test_files)) if test_files is not None else []
    test_removed = pd.DataFrame(columns=corpus_removed.columns)

    if supplied:
        test = load_named_streams(test_files, time_col, value_cols, stream_col)
        if show_progress:
            print("read {0} test stream(s)".format(len(test)))
        test, test_removed = screen_streams(test, show_progress=show_progress)
        test = lag_streams(test, lag, lag_method, show_progress=False)
        base = corpus
    else:
        if show_progress:
            print("no test files: cutting the test period off the corpus at base_fraction "
                  "{0}".format(base_fraction))
        base, test = [], []
        for name, frame in corpus:
            leading, trailing = cut_in_time(name, frame, base_fraction)
            base.append(leading)
            test.append(trailing)

    removed = pd.concat([corpus_removed, test_removed], ignore_index=True)
    channels = 0 if not base else sum(
        1 for column in base[0][1].columns if column != existing_time_name(base[0][1]))
    return base, test, {
        "corpus_streams_read": len(corpus) + len(corpus_removed),
        "corpus_streams_removed": len(corpus_removed),
        "test_streams_removed": len(test_removed),
        "test_period": "supplied" if supplied else "cut in time",
        "base_fraction": None if supplied else float(base_fraction),
        "lag": int(lag),
        "lag_method": lag_method if lag else None,
        "value_channels": channels,
        "removed": removed.to_dict(orient="records"),
    }


def write_summary(summary, summary_path, show_progress=True):
    """Write the stage's decisions beside the sets it produced."""
    if summary_path is None:
        return summary
    import json

    summary_path = Path(summary_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    if show_progress:
        print("wrote {0}".format(summary_path))
    return summary


def preprocess_streams(corpus_files, test_files, corpus_dir, test_dir, base_fraction=0.572,
                       time_col=None, value_cols=None, stream_col=None, lag=0,
                       lag_method='delay', summary_path=None, show_progress=True):
    """Stage entry point: two sets, `corpus` and `test`.

    One entry point for both workflows. They used to differ - the pooled one took a third set,
    cutting the corpus into `fit` and `withheld` so the threshold could be measured against
    normal data the metric had never seen. Cross-validated calibration made that redundant:
    k-fold over streams gives every interval a turn scored against an index built without its
    own stream, which is the same guarantee applied to the whole corpus rather than to a fifth
    of it. With that gone the two workflows want exactly the same thing from this stage.

    Returns
    -------
    dict
        Counts per set, how the test period was obtained, and the removal table as records.
    """
    corpus, test, summary = corpus_and_test(
        corpus_files, test_files, base_fraction, time_col, value_cols, stream_col, lag,
        lag_method, show_progress)

    counts = {"corpus": write_canonical_set(corpus, corpus_dir)[1],
              "test": write_canonical_set(test, test_dir)[1]}
    summary.update(counts)

    if show_progress:
        print("\ncorpus {corpus} / test {test} stream(s), {value_channels} value "
              "channel(s)".format(**dict(counts, **summary)))
        print("test period {0}".format(summary["test_period"]))
    return write_summary(summary, summary_path, show_progress)
