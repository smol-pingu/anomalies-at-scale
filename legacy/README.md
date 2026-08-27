# legacy

Superseded code, kept rather than deleted so the history of the approach stays readable.

## scripts/

The original standalone scripts, written before `src/anomalies_scale` existed. Every one
is superseded:

| legacy | replaced by |
|---|---|
| `scripts/signature_computer.py` | `anomalies_scale.signature_computer` |
| `scripts/score_streams.py` | `anomalies_scale.stream_scoring` |
| `scripts/nab_labels.py` | nothing - NAB-specific label parsing |

Note the name collision: the old `signature_computer.py` and the new module share a name
but nothing else. The old one composes signatures through Chen's identity and takes no
canonical input; the new one signs every interval directly and reads `(stream, Time,
Stream)`.

## notebooks/

`NAB_notebook.ipynb` was the only consumer of those scripts, and is superseded by
`notebooks/CMAPSS_notebook.ipynb`. It was already partly broken before being moved - it
imports `faiss_kNN`, which is not in this repository.

It sits beside `legacy/scripts/` so its relative imports still resolve if it is run from
here. `legacy/NAB-master/` was NOT moved: it is the NAB dataset, not code.
