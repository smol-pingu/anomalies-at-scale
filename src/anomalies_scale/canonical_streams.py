"""One canonical on-disk shape for a set of streams: one row per stream, not one file.

Every dataset this project touches arrives differently. SMD is a directory of headerless
comma-separated files, one per machine. C-MAPSS is space-separated text with a unit column
distinguishing 100 engines inside a single file. NAB is one CSV with a timestamp column. Each
of those shapes has been handled by a different code path, and every stage downstream has had
to know which it was looking at.

This module collapses them into one table::

    stream      Time                    Stream
    machine-1-1 [t0, t1, ...]           [[v0_0 ... v0_k], [v1_0 ... v1_k], ...]
    machine-1-2 [t0, t1, ...]           [[v0_0 ... v0_k], [v1_0 ... v1_k], ...]

`stream` identifies the stream, `Time` is its time coordinate, and `Stream` is its values as
one `(length, k)` block of ordered k-dimensional points.

Why time is a separate column
-----------------------------
It could have been folded in as channel 0 of `Stream`, which is how a path array is carried in
memory. Keeping it out has two advantages that matter on disk. The value points stay purely
k-dimensional, so `k` is the channel count and not the channel count plus one - which is the
number that goes into ``siglength`` and the number anyone reading the file wants to know. And
time keeps its own dtype and exactness: a timestamp column converted to elapsed seconds sits
badly next to values in the low single digits, and mixing them invites the scaling problem
that `preprocessing` exists to head off.

Streams need not be the same length. That is the point of storing them per row rather than as
one rectangular block - a fleet of engines with different lifetimes is the normal case, and
padding them to a common length would invent data.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------------------
# Reading streams off disk
#
# Absorbed from the old `create_corpus_merged`, which mixed file reading with corpus building
# and was dissolved when `signature_computer` took over the latter. Reading belongs here: this
# module is what every other stage goes through to turn a file into streams, and the canonical
# format exists precisely so that the rest of the pipeline never has to know which shape a
# dataset arrived in.
# ---------------------------------------------------------------------------------------

#: Column names checked, in order, when no ``time_col`` is given explicitly.
DEFAULT_TIME_COLUMNS = ('time', 'timestamp', 'datetime', 'date', 't')


def _looks_numeric(line, separator):
    """Whether every field of a line parses as a number.

    Used to decide whether a delimited text file has a header row. A first line that is
    entirely numeric is data, not names.
    """
    fields = [f for f in re.split(separator, line.strip()) if f != '']
    if not fields:
        return False
    for field in fields:
        try:
            float(field)
        except ValueError:
            return False
    return True


def read_text(path):
    """Read a plain ``.txt`` table, sniffing both the delimiter and whether it has a header.

    ``.txt`` is not one format. SMD ships comma-separated values with no header; C-MAPSS ships
    space-separated values with no header and a trailing space on every row that yields a
    phantom final column. Both are ordinary tables and both were previously unreadable, which
    meant the two datasets this project is actually built around could not enter the pipeline
    through :func:`read_frame` at all.

    Rather than add a configuration key per dataset, the delimiter is taken from whichever of
    comma / tab / whitespace splits the first line into the most fields, and a header is
    assumed only when that line is not entirely numeric.
    """
    path = Path(path)
    with path.open('r', encoding='utf-8', errors='replace') as handle:
        first = handle.readline()

    separators = {',': r',', '\t': r'\t', ' ': r'\s+'}
    separator = max(separators.values(),
                    key=lambda pattern: len(re.split(pattern, first.strip())))
    header = None if _looks_numeric(first, separator) else 'infer'

    frame = pd.read_csv(path, sep=separator, header=header, engine='python')
    return frame.dropna(axis=1, how='all')


#: File suffixes understood by :func:`load_stream`.
READERS = {
    '.parquet': pd.read_parquet,
    '.pq': pd.read_parquet,
    '.csv': pd.read_csv,
    '.tsv': lambda p: pd.read_csv(p, sep='\t'),
    '.txt': read_text,
    '.dat': read_text,
}


def read_frame(path):
    """Read one file into a DataFrame, dispatching on its suffix (see :data:`READERS`)."""
    path = Path(path)
    reader = READERS.get(path.suffix.lower())
    if reader is None:
        raise ValueError('unsupported file type {0!r} for {1}; expected one of {2}'.format(
            path.suffix, path.name, sorted(READERS)))
    return reader(path)


def resolve_time(frame, time_col=None):
    """Return a stream's time coordinate as a float array.

    Uses ``time_col`` if given, otherwise the first column whose name matches
    :data:`DEFAULT_TIME_COLUMNS`, otherwise the row index ``0..n-1``.

    Datetime-like columns are converted to seconds elapsed since the stream's own first
    timestamp, so each stream's time channel starts at zero and carries real units.

    Raises
    ------
    KeyError
        If ``time_col`` is given but absent from the frame - silently falling back to the
        row index there would quietly change what every signature means.
    """
    if time_col is not None:
        if time_col not in frame.columns:
            raise KeyError(
                'time column {0!r} not found; available columns: {1}'.format(
                    time_col, list(frame.columns)))
        column = frame[time_col]
    else:
        # Matched case-insensitively: preprocessing names the channel `Time`, and falling
        # back to the row index because the T was capitalised would silently change what
        # every signature in the run means.
        lowered = {str(name).lower(): name for name in frame.columns}
        column = None
        for candidate in DEFAULT_TIME_COLUMNS:
            if candidate in lowered:
                column = frame[lowered[candidate]]
                break
        if column is None:
            return np.arange(len(frame), dtype=float), None

    name = column.name

    if pd.api.types.is_numeric_dtype(column):
        return column.to_numpy(dtype=float), name

    # Timestamps commonly arrive as strings - pd.read_csv does not parse dates by default,
    # so a CSV round trip turns a datetime column into object dtype. Parse rather than
    # failing on it, since the alternative is a confusing float conversion error.
    if not pd.api.types.is_datetime64_any_dtype(column):
        try:
            column = pd.to_datetime(column)
        except (ValueError, TypeError) as error:
            raise ValueError(
                'time column {0!r} is neither numeric nor parseable as datetimes; '
                'first value {1!r}'.format(name, column.iloc[0])) from error

    seconds = (column - column.iloc[0]).dt.total_seconds()
    return seconds.to_numpy(dtype=float), name


def frame_to_path(frame, time_col=None, value_cols=None, source=''):
    """Convert one stream's rows into a ``(length, 1 + n_values)`` path array.

    The first column of the result is always time; the rest are the value channels, in frame
    order. Split out from :func:`load_stream` so a file holding several streams can be read
    once and then converted stream by stream.
    """
    time, time_name = resolve_time(frame, time_col)

    if value_cols is None:
        value_cols = [c for c in frame.columns if c != time_name]
    if not value_cols:
        raise ValueError('{0} has no value columns once {1!r} is removed'.format(
            source or 'frame', time_name))

    values = frame[list(value_cols)].to_numpy(dtype=float)
    return np.column_stack([time, values])


def load_stream(path, time_col=None, value_cols=None):
    """Read a single-stream file into a ``(length, 1 + n_values)`` path array.

    Parameters
    ----------
    path : str or Path
        File to read.
    time_col : str, optional
        Name of the time column. If omitted, :func:`resolve_time` looks for a conventional
        name and falls back to the row index.
    value_cols : sequence of str, optional
        Value columns to keep. Defaults to every column except the resolved time column.

    Returns
    -------
    np.ndarray
        Shape ``(length, 1 + n_values)``, dtype float.
    """
    return frame_to_path(read_frame(path), time_col, value_cols, source=Path(path).name)


def iter_streams(file, stream_col=None, time_col=None, value_cols=None):
    """Yield ``(name, path_array)`` for every stream held in one file.

    A file may carry one stream or many. With ``stream_col`` given, rows are grouped by that
    column and each group becomes its own stream, which is what lets a whole split arrive as
    a single long-format file rather than one file per window. Without it the file is one
    stream, which is the older behaviour and what one-file-per-window preprocessing produces.

    Names are qualified by the file stem, since stream identifiers only have to be unique
    within their own file but the corpus pools every stream from every file into one table.

    Raises
    ------
    KeyError
        If ``stream_col`` is given but absent from the file. Falling back to "the whole file
        is one stream" there would silently sign a concatenation of unrelated series as
        though it were a single path.
    """
    file = Path(file)
    frame = read_frame(file)
    stem = file.stem

    if stream_col is None:
        yield stem, frame_to_path(frame, time_col, value_cols, source=file.name)
        return

    if stream_col not in frame.columns:
        raise KeyError(
            'stream column {0!r} not found in {1}; available columns: {2}'.format(
                stream_col, file.name, list(frame.columns)))

    columns = value_cols
    for key, group in frame.groupby(stream_col, sort=True):
        rows = group.drop(columns=[stream_col]).reset_index(drop=True)
        yield ('{0}__{1}'.format(stem, key),
               frame_to_path(rows, time_col, columns, source=file.name))


def expand_paths(patterns):
    """Expand any wildcard arguments to matching files, sorted.

    Unix shells expand globs before the process sees them; PowerShell and cmd do not, so a
    pattern like ``data\\*.parquet`` would otherwise arrive as a literal filename and fail to
    open. Non-wildcard arguments pass through untouched.
    """
    import glob as _glob

    expanded = []
    for pattern in patterns:
        text = str(pattern)
        if any(ch in text for ch in '*?['):
            matches = sorted(_glob.glob(text))
            if not matches:
                raise FileNotFoundError('pattern {0!r} matched no files'.format(text))
            expanded.extend(matches)
        else:
            expanded.append(text)
    return expanded


#: The three columns, in order. `Stream` last because it is by far the widest.
STREAM_COLUMN = "stream"
TIME_COLUMN = "Time"
VALUES_COLUMN = "Stream"
CANONICAL_COLUMNS = (STREAM_COLUMN, TIME_COLUMN, VALUES_COLUMN)


def as_matrix(value):
    """Recover a ``(length, k)`` float array from whatever parquet handed back.

    Nested lists survive a parquet round trip as ragged object arrays of arrays rather than as
    a 2-D block, and the exact form depends on the engine. Normalising here means no caller
    has to care which engine wrote the file.
    """
    array = np.asarray(value, dtype=object) if isinstance(value, (list, tuple)) else value
    if isinstance(array, np.ndarray) and array.dtype == object:
        return np.asarray([np.asarray(row, dtype=float) for row in array], dtype=float)
    array = np.asarray(array, dtype=float)
    return array.reshape(len(array), -1) if array.ndim == 1 else array


def canonical_row(name, frame, time_col=None, value_cols=None):
    """Turn one stream's frame into a canonical record.

    Time resolves exactly as it does everywhere else in the project - via
    :func:`~anomalies_scale.canonical_streams.resolve_time` - so a datetime column becomes
    elapsed seconds from that stream's own first timestamp, and a file with no time channel
    falls back to its row index rather than being rejected.
    """
    time, time_name = resolve_time(frame, time_col)

    columns = [c for c in frame.columns if c != time_name]
    if value_cols:
        missing = [c for c in value_cols if c not in frame.columns]
        if missing:
            raise KeyError(
                "value column(s) {0} are not in stream {1!r}; it has {2}".format(
                    missing, name, list(frame.columns)))
        columns = [c for c in value_cols if c != time_name]
    if not columns:
        raise ValueError(
            "stream {0!r} has no value columns once the time column {1!r} is removed".format(
                name, time_name))

    return {
        STREAM_COLUMN: str(name),
        TIME_COLUMN: np.asarray(time, dtype=float),
        VALUES_COLUMN: frame[columns].to_numpy(dtype=float),
    }


def to_canonical(streams, time_col=None, value_cols=None):
    """Build the canonical frame from an iterable of ``(name, DataFrame)``."""
    rows = [canonical_row(name, frame, time_col, value_cols) for name, frame in streams]
    if not rows:
        return pd.DataFrame(columns=list(CANONICAL_COLUMNS))

    widths = {row[VALUES_COLUMN].shape[1] for row in rows}
    if len(widths) > 1:
        raise ValueError(
            "streams disagree about channel count ({0}); every stream in one set must have "
            "the same k, or nothing downstream can compare them".format(sorted(widths)))

    return pd.DataFrame(rows, columns=list(CANONICAL_COLUMNS))


def write_canonical(frame, path):
    """Write the canonical frame, converting the arrays to nested lists parquet can hold."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    storable = frame.copy()
    storable[TIME_COLUMN] = [np.asarray(t, dtype=float).tolist()
                             for t in storable[TIME_COLUMN]]
    storable[VALUES_COLUMN] = [as_matrix(v).tolist() for v in storable[VALUES_COLUMN]]
    storable.to_parquet(path, index=False)
    return path


def read_canonical(path):
    """Read a canonical file back, with the arrays restored."""
    inline = isinstance(path, pd.DataFrame)
    frame = path if inline else pd.read_parquet(Path(path))
    missing = [c for c in CANONICAL_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(
            "{0} is not a canonical stream set - it is missing {1}. Its columns are "
            "{2}".format("the frame given" if inline else path, missing,
                         list(frame.columns)[:8]))

    frame = frame.copy()
    frame[TIME_COLUMN] = [np.asarray(t, dtype=float) for t in frame[TIME_COLUMN]]
    frame[VALUES_COLUMN] = [as_matrix(v) for v in frame[VALUES_COLUMN]]
    return frame


def is_canonical(path):
    """Whether a file is a canonical stream set, cheaply and without reading its arrays.

    Lets a loader accept either a canonical set or the older one-stream-per-file layout
    without the caller having to declare which it is - the two coexist while the pipeline
    migrates, and a stage that guessed wrong would fail deep inside a reshape.
    """
    path = Path(path)
    if path.suffix.lower() not in {".parquet", ".pq"}:
        return False
    try:
        import pyarrow.parquet as pq

        columns = set(pq.read_schema(path).names)
    except Exception:
        try:
            columns = set(pd.read_parquet(path).columns)
        except Exception:
            return False
    return set(CANONICAL_COLUMNS).issubset(columns)


def iter_frames(source):
    """Yield ``(name, DataFrame)`` per stream, time and values as named columns.

    The frame form is what `preprocessing` works in - it names and screens columns - whereas
    :func:`iter_paths` gives the array form that signing needs. Same data, two consumers.
    """
    frame = read_canonical(source)
    for row in frame.itertuples(index=False):
        values = as_matrix(getattr(row, VALUES_COLUMN))
        block = pd.DataFrame(
            values, columns=["Variable_{0}".format(i + 1) for i in range(values.shape[1])])
        block.insert(0, TIME_COLUMN, np.asarray(getattr(row, TIME_COLUMN), dtype=float))
        yield str(getattr(row, STREAM_COLUMN)), block


def iter_paths(source):
    """Yield ``(name, path_array)`` for every stream, in the project's in-memory convention.

    On disk time is its own column; in memory the rest of this project carries a stream as one
    ``(length, 1 + k)`` array with time in column 0, because that is what ``iisignature`` is
    handed and what the interval slicing indexes. This is the join between the two, and it is
    the only place that has to know both.
    """
    frame = read_canonical(source)
    for row in frame.itertuples(index=False):
        time = np.asarray(getattr(row, TIME_COLUMN), dtype=float).reshape(-1, 1)
        values = as_matrix(getattr(row, VALUES_COLUMN))
        if len(time) != len(values):
            raise ValueError(
                "stream {0!r} has {1} timestamps but {2} points".format(
                    getattr(row, STREAM_COLUMN), len(time), len(values)))
        yield str(getattr(row, STREAM_COLUMN)), np.hstack([time, values])


def canonicalise_files(files, time_col=None, value_cols=None, stream_col=None,
                       show_progress=False):
    """Read raw files of any supported format into one canonical frame.

    A file may hold one stream or many; `stream_col` names the column that distinguishes them
    and is dropped once used, since it identifies a stream rather than measuring anything.
    Names are qualified by the file stem, because a stream identifier only has to be unique
    inside its own file while this table pools every file together.
    """
    streams = []
    for file in [Path(f) for f in files]:
        frame = read_frame(file)
        if stream_col is not None and stream_col in frame.columns:
            for key, group in frame.groupby(stream_col, sort=True):
                streams.append(("{0}__{1}".format(file.stem, key),
                                group.drop(columns=[stream_col]).reset_index(drop=True)))
        else:
            streams.append((file.stem, frame))

        if show_progress:
            print("read {0}".format(file.name))

    canonical = to_canonical(streams, time_col, value_cols)
    if show_progress and len(canonical):
        lengths = [len(t) for t in canonical[TIME_COLUMN]]
        print("{0} stream(s), k = {1}, lengths {2}-{3}".format(
            len(canonical), canonical[VALUES_COLUMN].iloc[0].shape[1],
            min(lengths), max(lengths)))
    return canonical


def summarise(frame):
    """A readable description of a canonical set: one row per stream, no arrays."""
    frame = read_canonical(frame)
    return pd.DataFrame({
        STREAM_COLUMN: frame[STREAM_COLUMN],
        "length": [len(t) for t in frame[TIME_COLUMN]],
        "k": [v.shape[1] for v in frame[VALUES_COLUMN]],
        "t_start": [float(t[0]) if len(t) else np.nan for t in frame[TIME_COLUMN]],
        "t_end": [float(t[-1]) if len(t) else np.nan for t in frame[TIME_COLUMN]],
        "finite": [bool(np.isfinite(v).all()) for v in frame[VALUES_COLUMN]],
    })
