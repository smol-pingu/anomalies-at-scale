"""Put the signature components of a corpus on comparable scales.

Signature terms are not commensurate with one another. A level-*k* term is a *k*-fold iterated
integral, so it scales roughly like the increment to the *k*-th power: on a path whose channels
move by ~1 per step the level-1 terms are order 1 and the level-2 terms order 1/2, and on one
whose channels move by ~100 the level-2 terms are four orders of magnitude larger. Nothing in
the signature transform corrects for that, and the raw corpus therefore has its variance
concentrated in whichever level happens to have the largest increments.

Why that matters even though the metric whitens
-----------------------------------------------
It is tempting to say the Mahalanobis whitening already handles scale, and for the *distance*
it does: ``Sigma^-1/2`` rescales every retained direction to unit variance regardless of what
it started at. What it does not fix is which directions get retained. ``variance_keep`` cuts by
share of total variance, so if level 2 carries 99.99% of the corpus variance simply by being
numerically larger, the retained rank is entirely level-2 directions and the level-1 structure
is discarded before the metric ever sees it. Normalising first makes that cut a statement about
information rather than about units.

Fit once, apply everywhere
--------------------------
This is a fitted transform, not a per-frame operation, and the distinction is the whole reason
it is a class. Normalising the corpus and the test set independently would give each its own
centre and scale, and a signature of a test interval would then be compared against corpus
signatures in different units - the same error as refitting a projection or a covariance on
test data. Fit on the corpus, keep the object, apply it to everything.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

#: Column prefix that marks a signature term. Everything else is provenance and passes through.
SIGNATURE_PREFIX = "sig_"

#: Available schemes. See :func:`fit_normaliser` for what each is for.
METHODS = ("zscore", "scale", "maxabs", "factorial")


def signature_columns(frame):
    """The signature columns of a corpus frame, in order, with the rest left alone."""
    columns = [c for c in frame.columns if str(c).startswith(SIGNATURE_PREFIX)]
    if not columns:
        raise ValueError(
            "no {0!r} columns found; expected a corpus from anomalies_scale."
            "signature_computer, got columns {1}".format(
                SIGNATURE_PREFIX, list(frame.columns)[:8]))
    return columns


def term_levels(n_terms, width):
    """Which signature level each term belongs to, for a path of `width` channels.

    ``iisignature`` lays terms out level by level - ``width`` at level 1, ``width**2`` at
    level 2, and so on, with the constant term omitted - so the levels are recoverable from the
    term count alone once the width is known.
    """
    levels, level, size = [], 1, width
    while len(levels) < n_terms:
        levels.extend([level] * min(size, n_terms - len(levels)))
        level += 1
        size *= width
    if len(levels) != n_terms:
        raise ValueError(
            "{0} term(s) is not a whole number of levels for width {1}".format(
                n_terms, width))
    return np.asarray(levels, dtype=int)


class Normaliser:
    """A fitted per-component normalisation: a centre and a scale per signature term.

    ``transform`` returns a new frame with the signature columns rescaled and every other
    column - ``depth``, ``stream``, ``lo``, ``hi``, ``n_cells`` - carried through untouched, in
    its original position. Those identify which interval a row is; rescaling them would be
    meaningless and dropping them would lose the provenance the corpus is built to keep.
    """

    def __init__(self, columns, centre, scale, method, params=None):
        self.columns = list(columns)
        self.centre = np.asarray(centre, dtype=float)
        self.scale = np.asarray(scale, dtype=float)
        self.method = str(method)
        self.params = dict(params or {})

    @property
    def n_constant(self):
        """How many terms carried no variance in the corpus and are therefore passed to zero."""
        return int((self.scale == 1.0).sum() if self.method != "factorial"
                   else 0)

    def transform(self, frame):
        """Apply the fitted normalisation to a corpus frame."""
        columns = signature_columns(frame)
        if columns != self.columns:
            raise ValueError(
                "this normaliser was fitted on {0} term(s) beginning {1}, but the frame has "
                "{2} beginning {3}. A signature normalised against the wrong terms is not "
                "comparable to the corpus.".format(
                    len(self.columns), self.columns[:3], len(columns), columns[:3]))

        out = frame.copy()
        values = frame[columns].to_numpy(dtype=float)
        out[columns] = (values - self.centre) / self.scale
        return out

    def inverse_transform(self, frame):
        """Undo the normalisation, for reading a value back in its original units."""
        columns = signature_columns(frame)
        out = frame.copy()
        out[columns] = frame[columns].to_numpy(dtype=float) * self.scale + self.centre
        return out

    def to_dict(self):
        """The fitted parameters, as something JSON can hold."""
        return {"method": self.method, "columns": self.columns,
                "centre": self.centre.tolist(), "scale": self.scale.tolist(),
                "params": self.params}

    def save(self, path):
        """Write the fitted parameters as readable JSON.

        JSON rather than pickle: this is two float vectors and a method name, it should stay
        readable when a version changes, and nothing here holds compiled state.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path

    @staticmethod
    def load(path):
        """Read back a normaliser written by :meth:`save`."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return Normaliser(data["columns"], data["centre"], data["scale"], data["method"],
                          data.get("params"))

    def __repr__(self):
        return "<Normaliser {0} over {1} term(s)>".format(self.method, len(self.columns))


def fit_normaliser(frame, method="zscore", width=None):
    """Fit a per-component normalisation on a corpus frame.

    Parameters
    ----------
    method : str
        ``'zscore'``
            Centre each term and divide by its standard deviation. The default, and the one
            that makes ``variance_keep`` a statement about information rather than units.
        ``'scale'``
            Divide by the standard deviation without centring. Keeps the origin where it is,
            which matters if anything downstream reads a signature's absolute position rather
            than only distances between signatures.
        ``'maxabs'``
            Divide by the largest absolute value. Bounds every term to [-1, 1] without
            assuming the corpus is roughly symmetric about its mean.
        ``'factorial'``
            Divide each level-*d* term by ``d!``, the classical signature rescaling. Corrects
            the systematic decay of deep terms analytically rather than empirically, so it
            needs no corpus statistics and is unaffected by a small or skewed corpus - but it
            does nothing about channels being on different scales. Requires `width`.
    width : int, optional
        Path channel count, needed only by ``'factorial'`` to work out which level each term
        belongs to.

    Zero-variance terms
    -------------------
    A term that never moves in the corpus has no scale to divide by. Its scale is set to 1 so
    that centring sends it to exactly zero rather than to a NaN or an infinity. That is the
    right answer: a constant direction carries no information, and the whitening's pseudo-
    inverse would annihilate it anyway.
    """
    if method not in METHODS:
        raise ValueError("method must be one of {0}, got {1!r}".format(METHODS, method))

    columns = signature_columns(frame)
    values = frame[columns].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(
            "the corpus carries non-finite signature values; a normalisation fitted on them "
            "would propagate into every transformed frame")

    if method == "factorial":
        if width is None:
            raise ValueError("method='factorial' needs `width`, the path's channel count")
        levels = term_levels(len(columns), int(width))
        centre = np.zeros(len(columns))
        scale = np.array([float(math.factorial(int(d))) for d in levels])
        params = {"width": int(width), "levels": levels.tolist()}
        return Normaliser(columns, centre, scale, method, params)

    if method == "maxabs":
        centre = np.zeros(len(columns))
        scale = np.abs(values).max(axis=0)
    else:
        centre = values.mean(axis=0) if method == "zscore" else np.zeros(len(columns))
        scale = values.std(axis=0)

    constant = scale <= 0
    scale = np.where(constant, 1.0, scale)

    return Normaliser(columns, centre, scale, method,
                      {"n_constant_terms": int(constant.sum()), "n_rows_fitted": len(values)})


def normalise_corpus(frame, method="zscore", width=None, output_path=None,
                     normaliser_path=None, show_progress=False):
    """Fit on `frame` and return it normalised, alongside the fitted normaliser.

    The normaliser is returned as well as the frame, and deliberately so: without it the test
    side cannot be put into the same units, and a corpus normalised on its own statistics is
    not comparable with anything.

    Returns
    -------
    (pd.DataFrame, Normaliser)
    """
    normaliser = fit_normaliser(frame, method=method, width=width)
    out = normaliser.transform(frame)

    if show_progress:
        kept = [c for c in out.columns if not str(c).startswith(SIGNATURE_PREFIX)]
        print("normalised {0} term(s) by {1!r} over {2:,} row(s); {3} column(s) passed "
              "through: {4}".format(len(normaliser.columns), method, len(frame),
                                    len(kept), ", ".join(str(c) for c in kept)))
        constant = normaliser.params.get("n_constant_terms")
        if constant:
            print("  {0} term(s) never moved in the corpus and are now exactly zero".format(
                constant))

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        out.to_parquet(output_path, index=False)
    if normaliser_path is not None:
        normaliser.save(normaliser_path)

    return out, normaliser
