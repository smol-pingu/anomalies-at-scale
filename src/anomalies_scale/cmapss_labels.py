"""Derive per-cycle ground truth for C-MAPSS, in the convention the notebook uses.

C-MAPSS is a prognostics benchmark, not an anomaly-detection one, and the difference shows up
in what ships as ground truth. There are no point labels at all: the run-to-failure half has
none, and the truncated half comes with a single integer per engine - the cycles it would have
survived past the end of its record. Point labels have to be declared.

The convention
--------------
The piecewise-linear RUL target standard in the prognostics literature holds that sensor
response to damage is not considered observable until roughly the last 125-130 cycles, and
treats everything above that knee as flat health. Following it, a cycle is called degraded when
its remaining useful life is at or below ``horizon``.

``horizon`` is a *choice*, not something the data states, and it is worth being explicit that
moving it moves the answer rather than revealing it. Measured on this data, taking the horizon
from 30 to 50 raised precision from 0.721 to 0.942 without the detector changing at all - the
same flags, rescored against a different convention.

Which half this applies to
--------------------------
The run-to-failure half only, where remaining life is exact: the record ends at failure, so RUL
at any cycle is simply what is left of it. In the truncated half RUL depends on
``RUL_FDxxx.txt`` being right, and the last cycle is *not* failure - which is why this project
scores the run-to-failure half and builds its corpus from the truncated one.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from anomalies_scale.canonical_streams import STREAM_COLUMN, read_canonical

#: Cycles-to-failure at or below which a cycle is called degraded.
DEFAULT_HORIZON = 30

#: Column name `stream_evaluation` reads a per-point ground truth from.
LABEL_COLUMN = "label"


def labels_for_run_to_failure(streams, horizon=DEFAULT_HORIZON, output_path=None,
                             show_progress=False):
    """Per-cycle labels for a canonical set of run-to-failure streams.

    The record ends at failure, so remaining life at position *i* of an *L*-cycle engine is
    ``L - 1 - i``, and the degraded cycles are exactly the last ``horizon + 1``. Derived from
    the stream lengths rather than by re-reading the raw file, so the labels cannot drift out
    of alignment with the streams they describe - which is the failure that would otherwise be
    silent, since a shifted label array is still the right length.

    Returns
    -------
    pd.DataFrame
        ``stream`` and ``label``: a 0/1 array the length of each stream.
    """
    frame = read_canonical(streams)

    rows = []
    for row in frame.itertuples(index=False):
        length = len(np.asarray(getattr(row, "Time")))
        remaining = np.arange(length - 1, -1, -1)
        rows.append({STREAM_COLUMN: str(getattr(row, STREAM_COLUMN)),
                     LABEL_COLUMN: (remaining <= horizon).astype(int)})

    labels = pd.DataFrame(rows)

    if show_progress:
        total = int(sum(row[LABEL_COLUMN].sum() for row in rows))
        points = int(sum(len(row[LABEL_COLUMN]) for row in rows))
        carrying = int(sum(1 for row in rows if row[LABEL_COLUMN].any()))
        print("{0} stream(s), {1:,} point(s); {2:,} degraded at RUL <= {3} ({4:.2%}), "
              "{5}/{0} stream(s) carry a failure".format(
                  len(rows), points, total, horizon, total / points, carrying))

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        storable = labels.copy()
        storable[LABEL_COLUMN] = [array.tolist() for array in storable[LABEL_COLUMN]]
        storable.to_parquet(output_path, index=False)
        if show_progress:
            print("wrote {0}".format(output_path))

    return labels
