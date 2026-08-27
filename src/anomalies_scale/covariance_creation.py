"""Fit the Mahalanobis metric for a signature corpus and write it out as a CSV matrix.

Takes a corpus from :mod:`anomalies_scale.signature_computer`, forms the covariance of its
signature vectors, and writes the (pseudo-)inverse as a dense ``D x D`` matrix carrying the
signature term names. Keeping the names is not decoration: the matrix is only meaningful
against columns in the same order, and a corpus rebuilt at a different truncation would
otherwise line up silently and wrongly.

How it is stored
----------------
``npz`` by default, ``csv`` on request; the format follows the output path's suffix, so a
caller chooses by naming the file. The matrix is dense float64 and its size grows with the
*square* of the dimension, which is what makes the choice matter::

    terms      entries        csv       npz
      650        423 K     8.8 MB    3.4 MB
    5,525       30.5 M     ~600 MB    244 MB     log-signature at truncation 3
   16,275        265 M       ~5 GB    2.1 GB     signature at truncation 3

At 650 terms this is a rounding error and either would do. At truncation 3 the decimal-text
form takes minutes to parse on every stage that reads it, which is a blocker rather than an
inconvenience - so the default is the one that still works at the dimensions this pipeline is
heading toward. ``csv`` remains readable for anything that wants the matrix without numpy, and
so that matrices already on disk keep loading.

Which matrix to write
---------------------
Two forms are available, and they are *not* interchangeable. Both derive from one SVD of the
centred corpus, differing only in the exponent applied to the singular values.

``pinv`` (the default)
    ``Sigma^-1``, the Moore-Penrose pseudo-inverse of the covariance. This is the matrix in
    the textbook Mahalanobis formula, to be used as a quadratic form::

        d2 = (x - y) @ M @ (x - y).T

``inv_sqrt``
    ``Sigma^-1/2``, the symmetric square root of the above, which turns Mahalanobis distance
    into plain Euclidean distance between transformed vectors::

        d2 = ||(x - y) @ W|| ** 2

The distinction matters because it is easy to get wrong and expensive when you do. Passing a
``pinv`` matrix into a whitening-then-Euclidean path - which is what a FAISS ``IndexFlatL2``
does - computes ``(x - y) @ Sigma^-2 @ (x - y).T``, weighting every direction quadratically.
On data as ill-conditioned as signature vectors that is not a small error: the toy notebook
measured the two formulas disagreeing by a factor of ~5e9, and it was the difference between
localising two anomalies and missing them entirely. Pick ``inv_sqrt`` for anything that
indexes with Euclidean distance, ``pinv`` for an explicit quadratic form.

Rank
----
Signature corpora are nowhere near full rank in any practical sense - on SMD, 56 of 992
directions carry 99.9% of the variance - and the remaining directions describe sampling noise
in this particular corpus. Inverting them multiplies that noise by an enormous factor, so the
spectrum is truncated by *retained variance* rather than by a fixed singular-value floor,
which adapts to the corpus instead of assuming its conditioning.

Numerically the fit never forms ``X^T X``, which would square an already-brutal condition
number, and never materialises the corpus as one float64 matrix: it folds centred chunks into
a running QR factor, exactly as :func:`anomaly_detection_pooled.build_whitening` does.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

#: Matrix forms this module can emit, and the exponent each applies to the covariance.
FORMS = ("pinv", "inv_sqrt")

#: How a corpus is read, dispatched on suffix - matching what signature_computer writes.
CORPUS_READERS = {
    ".parquet": pd.read_parquet,
    ".pq": pd.read_parquet,
    ".csv": pd.read_csv,
}

#: Suffixes the fitted matrix may be written with. See the module docstring for the trade.
MATRIX_FORMATS = (".npz", ".csv")


def write_covariance(labelled, output_path, whitening=None):
    """Write a fitted matrix, dispatching on the output path's suffix.

    Given a `whitening` with its factor, writes ``basis`` and ``scale`` instead of ``matrix`` -
    7,150 numbers rather than 422,500 on C-MAPSS, and the form `load_covariance` needs to index
    in latent coordinates. Only ``npz`` can carry it; a csv request falls back to the dense
    matrix, since a labelled table of the factor would not be the readable thing csv is for.

    ``np.savez`` rather than ``savez_compressed``: the matrix is dense float64 with little
    structure for zlib to find, so compression buys around a tenth of the size and costs CPU on
    a file that `index`, `calibrate` and `score` each read once per run. The subspace basis
    written beside it *is* compressed, because it is written once and read once.

    Both forms carry the term names, and :func:`load_covariance` checks them either way.
    """
    output_path = Path(output_path)
    suffix = output_path.suffix.lower()
    if suffix not in MATRIX_FORMATS:
        raise ValueError(
            "unsupported matrix format {0!r} for {1}; expected one of {2}".format(
                output_path.suffix, output_path.name, list(MATRIX_FORMATS)))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if suffix == ".csv":
        labelled.to_csv(output_path)
    elif whitening is not None and whitening.factored:
        np.savez(output_path,
                 basis=whitening.basis.astype(np.float64),
                 scale=whitening.scale.astype(np.float64),
                 columns=np.asarray(labelled.columns, dtype=str))
    else:
        np.savez(output_path, matrix=labelled.to_numpy(dtype=np.float64),
                 columns=np.asarray(labelled.columns, dtype=str))
    return output_path


def read_corpus(source):
    """Load a signature corpus, dispatching on file suffix.

    Accepts an already-loaded DataFrame so a corpus can be built and fitted in one session
    without a round trip through disk.
    """
    if isinstance(source, pd.DataFrame):
        return source

    path = Path(source)
    reader = CORPUS_READERS.get(path.suffix.lower())
    if reader is None:
        raise ValueError('unsupported corpus format {0!r} for {1}; expected one of {2}'.format(
            path.suffix, path.name, sorted(CORPUS_READERS)))
    return reader(path)


def signature_matrix(frame, show_progress=False):
    """Separate a corpus frame into its signature matrix and the names of those columns.

    A corpus from :mod:`anomalies_scale.signature_computer` carries provenance alongside the
    terms - ``depth``, ``stream``, ``lo``, ``hi``, ``n_cells`` - and every one of those is
    dropped here. They identify *which* interval a row is, not what it looks like, and folding
    an interval index or a stream name into a covariance would have the metric measure the
    corpus's own bookkeeping.

    The ``sig_`` columns must be in numeric order, because the matrix this feeds is written out
    labelled by these names and applied positionally later. A frame whose columns had been
    reordered - by a merge or a column selection - would produce a matrix that looks right,
    inverts fine, and computes distances in a permuted basis.
    """
    columns = [c for c in frame.columns if c.startswith("sig_")]
    if not columns:
        raise ValueError(
            "no 'sig_' columns found; expected a corpus from anomalies_scale."
            "signature_computer, got columns {0}".format(list(frame.columns)[:8]))

    try:
        indices = [int(name.split("_", 1)[1]) for name in columns]
    except ValueError:
        raise ValueError(
            "signature columns are not named 'sig_<n>': got {0}".format(columns[:6]))
    if indices != list(range(len(columns))):
        raise ValueError(
            "signature columns are out of order or have gaps - they run {0}..{1} in {2} "
            "column(s). The covariance is written labelled by these names and applied "
            "positionally, so a permuted corpus would give distances in a permuted basis."
            .format(indices[0], indices[-1], len(indices)))

    dropped = [c for c in frame.columns if not c.startswith("sig_")]
    if show_progress and dropped:
        print("dropping {0} non-signature column(s): {1}".format(
            len(dropped), ", ".join(str(c) for c in dropped)))

    return frame[columns].to_numpy(dtype=float), columns


def spectrum(signatures, chunk_rows=200_000):
    """Singular values and right singular vectors of the centred corpus.

    Two streaming passes - one for the mean, one folding centred chunks into a running QR
    factor. Stacking ``[R_prev; chunk]`` and re-factorising is exact: R carries the same
    singular values and right singular vectors as the full stacked matrix, so this matches a
    direct fit to machine precision at ``O(chunk_rows * D)`` memory.

    Returns
    -------
    (np.ndarray, np.ndarray, int)
        Singular values, ``Vt``, and the number of rows fitted.
    """
    signatures = np.asarray(signatures)
    n_rows, n_terms = signatures.shape
    if n_rows < 2:
        raise ValueError("need at least 2 corpus intervals to fit a covariance, got {0}"
                         .format(n_rows))

    total = np.zeros(n_terms, dtype=np.float64)
    for start in range(0, n_rows, chunk_rows):
        total += signatures[start:start + chunk_rows].sum(axis=0, dtype=np.float64)
    mean = total / n_rows

    factor = np.zeros((0, n_terms), dtype=np.float64)
    for start in range(0, n_rows, chunk_rows):
        chunk = np.asarray(signatures[start:start + chunk_rows], dtype=np.float64) - mean
        factor = np.linalg.qr(np.vstack([factor, chunk]), mode="r")

    _, values, right = np.linalg.svd(factor, full_matrices=False)
    return values, right, n_rows


def retained_rank(values, variance_keep):
    """How many leading components carry `variance_keep` of the corpus's variance.

    Never exceeds the numerical rank, so a corpus that is genuinely degenerate is not handed
    directions that are pure rounding error.
    """
    if values.size == 0 or values.max() <= 0:
        raise ValueError("corpus has no variance in any direction")

    numerical_rank = int((values > 1e-12 * values.max()).sum())
    if variance_keep is None:
        return numerical_rank

    energy = np.cumsum(values ** 2) / np.sum(values ** 2)
    return min(int(np.searchsorted(energy, variance_keep) + 1), numerical_rank)


def numerical_subspace(signatures, rcond=1e-12):
    """Orthonormal basis of the corpus's numerically non-zero row space.

    This is **not** the subspace the metric keeps. `variance_keep` cuts by share of variance and
    on C-MAPSS leaves 11 directions of 650; this cuts only what is numerically zero and leaves
    a few hundred. The two answer different questions, and the off-manifold test needs this one:
    a direction the corpus has *little* variance in is unusual, but a direction it has *no*
    variance in is one the corpus cannot represent at all.

    Testing against the variance-truncated subspace instead would call almost every test
    interval off-manifold, since the discarded directions are ordinary corpus variation.

    Returns
    -------
    np.ndarray
        ``(rank, n_terms)``, rows orthonormal.
    """
    values, right, _ = spectrum(signatures)
    if values.size == 0 or values.max() <= 0:
        raise ValueError("corpus has no variance in any direction")
    rank = int((values > rcond * values.max()).sum())
    return right[:rank]


class Whitening:
    """The fitted metric, as either a dense matrix or the rank-r factor behind it.

    ``covariance_matrix`` builds the whitening as ``W = V_r.T diag(s) V_r`` with ``V_r`` of
    shape ``(r, D)`` and orthonormal rows, then multiplies it out into a ``D x D`` array. On
    C-MAPSS that is a 650 x 650 matrix of rank 8 to 11 - 422,500 numbers expressing something
    that needs 7,150.

    Keeping the factor is not only smaller. Because ``V_r`` has orthonormal rows, the latent
    coordinates ``z = x V_r.T diag(s)`` are an **exact isometry** of the whitened vectors::

        x_w = z V_r          =>      ||x_w - y_w||^2
                                       = (z_x - z_y) V_r V_r.T (z_x - z_y).T
                                       = ||z_x - z_y||^2                 since V_r V_r.T = I_r

    So a FAISS index over *r* coordinates returns identical neighbours and identical distances
    to one over *D*, at r/D of the memory and the search cost. On FD001 that is 76.1 MB of index
    against 1.29 MB; on FD004, 240 MB against 4 MB. The claim is exact rather than approximate,
    and is asserted in `latent_isometry_error` rather than taken on trust.

    ``dense`` reconstructs the ``D x D`` form for the one caller that genuinely needs it -
    `index_creation.check_is_whitening`, which tests that whitening leaves the corpus covariance
    a projection and so has to see the matrix itself.
    """

    def __init__(self, basis, scale, columns=None, matrix=None):
        #: ``(r, D)`` orthonormal rows, or None when only a dense matrix is held.
        self.basis = None if basis is None else np.asarray(basis, dtype=float)
        self.scale = None if scale is None else np.asarray(scale, dtype=float)
        self._matrix = None if matrix is None else np.asarray(matrix, dtype=float)
        self.columns = list(columns) if columns is not None else None
        if self.basis is None and self._matrix is None:
            raise ValueError("a Whitening needs either a factor or a dense matrix")

    @classmethod
    def from_dense(cls, matrix, columns=None):
        """Wrap a matrix that was stored densely - every artifact written before the factor."""
        return cls(None, None, columns, matrix)

    @property
    def factored(self):
        return self.basis is not None

    @property
    def dimension(self):
        """Signature terms the metric was fitted on - *D*, never the latent width.

        Distinct from `latent` on purpose: `stream_scoring` infers the truncation level from the
        number of signature terms, and would infer the wrong one from a latent width.
        """
        return int(self.basis.shape[1] if self.factored else self._matrix.shape[0])

    @property
    def latent(self):
        """Columns `apply` returns - *r* when factored, *D* when dense."""
        return int(self.basis.shape[0] if self.factored else self._matrix.shape[0])

    def apply(self, signatures):
        """Put signature vectors into the space distances are measured in."""
        signatures = np.atleast_2d(np.asarray(signatures, dtype=float))
        if not self.factored:
            return signatures @ self._matrix.T
        return (signatures @ self.basis.T) * self.scale

    def dense(self):
        """The ``D x D`` matrix, reconstructed from the factor if that is what is held."""
        if self._matrix is None:
            self._matrix = (self.basis.T * self.scale) @ self.basis
        return self._matrix

    def latent_isometry_error(self, signatures, sample=200, random_state=0):
        """Largest relative disagreement between latent and full-width distances.

        The whole justification for indexing *r* coordinates instead of *D*. Checked rather than
        assumed, on a sample, because a silent failure here would produce plausible distances in
        the wrong geometry - the same class of error `check_is_whitening` exists to catch.
        """
        if not self.factored:
            return 0.0
        rows = np.random.default_rng(random_state).choice(
            len(signatures), size=min(int(sample), len(signatures)), replace=False)
        block = np.asarray(signatures, dtype=float)[rows]
        latent = self.apply(block)
        full = block @ self.dense().T
        near = np.linalg.norm(latent[:, None] - latent[None], axis=-1)
        far = np.linalg.norm(full[:, None] - full[None], axis=-1)
        scale = float(far.max())
        return float(np.abs(near - far).max() / scale) if scale else 0.0


def covariance_matrix(signatures, variance_keep=0.999, form="pinv"):
    """Fit the covariance of `signatures` and return the requested inverse form.

    Parameters
    ----------
    signatures : np.ndarray
        ``(n_intervals, n_terms)`` corpus of signature vectors.
    variance_keep : float, optional
        Fraction of signature variance the retained directions must carry. ``None`` keeps
        every numerically non-zero direction, which is rarely what you want - see the module
        docstring.
    form : str
        ``'pinv'`` for ``Sigma^-1``, ``'inv_sqrt'`` for ``Sigma^-1/2``.

    Returns
    -------
    (np.ndarray, dict)
        The ``(n_terms, n_terms)`` matrix, and a dict of diagnostics: the rank kept, the
        dimension, the rows fitted, and the variance actually retained.
    """
    if form not in FORMS:
        raise ValueError("form must be one of {0}, got {1!r}".format(list(FORMS), form))

    values, right, n_rows = spectrum(signatures)
    rank = retained_rank(values, variance_keep)

    # Sigma = V diag(s**2 / n) V.T, so the two forms differ only in this exponent.
    if form == "pinv":
        scale = n_rows / values[:rank] ** 2
    else:
        scale = np.sqrt(n_rows) / values[:rank]

    kept = right[:rank]
    matrix = (kept.T * scale) @ kept
    factor = Whitening(kept, scale, matrix=matrix)

    energy = float(np.sum(values[:rank] ** 2) / np.sum(values ** 2))
    diagnostics = {
        "form": form,
        "dimension": int(signatures.shape[1]),
        "n_intervals": int(n_rows),
        "rank": int(rank),
        "variance_retained": energy,
        "condition_number": float(values[0] / values[rank - 1]),
    }
    # The dense matrix stays the return value, so every existing caller is unaffected. The
    # factor rides along on it for the ones that want to index in latent coordinates.
    diagnostics["whitening"] = factor
    return matrix, diagnostics


def create_covariance(corpus, output_path=None, variance_keep=0.999, form="pinv",
                      diagnostics_path=None, subspace_path=None, factored=False,
                      show_progress=False):
    """Fit the metric for one corpus and write it as a CSV matrix.

    The CSV carries the signature term names as both its header and its index, so the matrix
    cannot be silently applied to a corpus whose columns are in a different order.

    Returns
    -------
    (pd.DataFrame, dict)
        The matrix as a labelled frame, and its diagnostics.
    """
    frame = read_corpus(corpus)
    signatures, columns = signature_matrix(frame, show_progress=show_progress)

    matrix, info = covariance_matrix(signatures, variance_keep=variance_keep, form=form)
    labelled = pd.DataFrame(matrix, index=columns, columns=columns)
    labelled.index.name = "term"

    # Recorded so a matrix can be traced back to the corpus it came from, and so that a run
    # fitted on a subset of depths is distinguishable afterwards from one fitted on all of them.
    info["n_columns_dropped"] = int(len(frame.columns) - len(columns))
    if "depth" in frame.columns:
        info["depths"] = sorted(int(d) for d in frame["depth"].unique())
    if "stream" in frame.columns:
        info["n_streams"] = int(frame["stream"].nunique())

    if subspace_path is not None:
        # A second decomposition rather than one shared with `covariance_matrix`. The two want
        # different cuts of the same spectrum, and at these sizes a QR plus SVD costs seconds -
        # not worth threading a second return value through every caller to save.
        basis = numerical_subspace(signatures)
        info["numerical_rank"] = int(basis.shape[0])

        subspace_path = Path(subspace_path)
        subspace_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(subspace_path, basis=basis.astype(np.float64),
                            columns=np.asarray(columns, dtype=str))
        if show_progress:
            print("numerical rank {0} of {1}; wrote the off-manifold basis to {2}".format(
                basis.shape[0], info["dimension"], subspace_path))

    if show_progress:
        print("fitted {form} on {n_intervals:,} intervals x {dimension} terms: rank "
              "{rank}, retaining {variance_retained:.4%} of variance".format(**info))

    whitening = info.pop("whitening")
    whitening.columns = list(columns)

    if factored:
        # The claim the latent index rests on, measured rather than assumed: distances between
        # r-dimensional coordinates must equal distances between D-dimensional whitened vectors.
        info["latent_dimension"] = int(whitening.latent)
        info["latent_isometry_error"] = whitening.latent_isometry_error(signatures)
        if info["latent_isometry_error"] > 1e-6:
            raise ValueError(
                "the rank-{0} factor does not reproduce full-width distances (relative error "
                "{1:.2e}). Something is wrong with the decomposition; refusing to index in a "
                "geometry that is not the one the metric describes."
                .format(whitening.latent, info["latent_isometry_error"]))

    if output_path is not None:
        output_path = write_covariance(labelled, output_path,
                                       whitening=whitening if factored else None)
        if show_progress:
            size = output_path.stat().st_size / 1024 ** 2
            if factored:
                print("wrote the rank-{0} factor of a {1} x {1} matrix to {2} ({3:.2f} MB; "
                      "dense would be {4:.2f} MB), isometry error {5:.1e}".format(
                          whitening.latent, info["dimension"], output_path, size,
                          info["dimension"] ** 2 * 8 / 1024 ** 2,
                          info["latent_isometry_error"]))
            else:
                print("wrote {0} x {0} matrix to {1} ({2:.1f} MB)".format(
                    info["dimension"], output_path, size))

    if diagnostics_path is not None:
        diagnostics_path = Path(diagnostics_path)
        diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
        with open(diagnostics_path, "w", encoding="utf-8") as handle:
            json.dump(info, handle, indent=2)

    return labelled, info


def load_covariance(path, expected_columns=None):
    """Read a matrix written by :func:`create_covariance`, checking it lines up.

    Dispatches on suffix, so ``npz`` and ``csv`` are both accepted and a matrix fitted before
    this module gained the binary form still loads.

    Parameters
    ----------
    path : str or Path
        ``npz`` or ``csv`` written by this module.
    expected_columns : sequence of str, optional
        Signature column names the matrix is about to be applied to. Checked against the
        matrix's own labels, since a mismatch would otherwise produce a plausible-looking
        distance computed in the wrong basis.
    """
    path = Path(path)
    factor = None
    if path.suffix.lower() == ".csv":
        frame = pd.read_csv(path, index_col=0)
        matrix, columns = frame.to_numpy(dtype=float), list(frame.columns)
    elif path.suffix.lower() == ".npz":
        stored = np.load(path, allow_pickle=False)
        columns = [str(name) for name in stored["columns"]]
        if "basis" in stored.files:
            # The rank-r factor. Kept factored rather than multiplied out, so callers can
            # project into latent coordinates instead of the full width.
            factor = (np.asarray(stored["basis"], dtype=float),
                      np.asarray(stored["scale"], dtype=float))
            matrix = None
        else:
            matrix = np.asarray(stored["matrix"], dtype=float)
    else:
        raise ValueError(
            "unsupported matrix format {0!r} for {1}; expected one of {2}".format(
                path.suffix, path.name, list(MATRIX_FORMATS)))

    if expected_columns is not None and columns != list(expected_columns):
        raise ValueError(
            "covariance at {0} does not match the corpus it is being applied to: it has "
            "{1} terms, the corpus has {2}, and the first difference is at position {3}"
            .format(path, len(columns), len(expected_columns),
                    next((i for i, (a, b) in enumerate(zip(columns, expected_columns))
                          if a != b), min(len(columns), len(expected_columns)))))
    return matrix


def load_whitening(path, expected_columns=None):
    """The fitted metric as a :class:`Whitening`, whichever form it was stored in.

    Callers that only want to multiply should use this and call ``apply``; it keeps the factor
    when there is one, so the multiplication is ``2 N D r`` instead of ``N D^2`` and the result
    is *r* columns rather than *D*. A matrix stored densely - anything written before the factor
    existed - loads as a dense `Whitening` and behaves exactly as it always did.
    """
    path = Path(path)
    if path.suffix.lower() == ".npz":
        stored = np.load(path, allow_pickle=False)
        columns = [str(name) for name in stored["columns"]]
        _check_columns(path, columns, expected_columns)
        if "basis" in stored.files:
            return Whitening(np.asarray(stored["basis"], dtype=float),
                             np.asarray(stored["scale"], dtype=float), columns)
        return Whitening.from_dense(np.asarray(stored["matrix"], dtype=float), columns)

    return Whitening.from_dense(load_covariance(path, expected_columns))


def _check_columns(path, columns, expected_columns):
    if expected_columns is None or columns == list(expected_columns):
        return
    raise ValueError(
        "covariance at {0} does not match the corpus it is being applied to: it has {1} "
        "terms, the corpus has {2}, and the first difference is at position {3}"
        .format(path, len(columns), len(expected_columns),
                next((i for i, (a, b) in enumerate(zip(columns, expected_columns))
                      if a != b), min(len(columns), len(expected_columns)))))


def main(argv=None):
    """Command line entry point."""
    parser = argparse.ArgumentParser(
        description="Fit the Mahalanobis metric for a signature corpus.")
    parser.add_argument("corpus", help="corpus parquet or csv from signature_computer")
    parser.add_argument("--output", required=True,
                        help="where to write the matrix; the suffix picks the format, "
                             "npz (default elsewhere in the workflow) or csv")
    parser.add_argument("--form", choices=FORMS, default="pinv",
                        help="pinv for Sigma^-1 (quadratic form), inv_sqrt for Sigma^-1/2 "
                             "(whitening, for Euclidean indexes); default pinv")
    parser.add_argument("--variance-keep", type=float, default=0.999,
                        help="fraction of signature variance the retained rank must carry")
    parser.add_argument("--diagnostics", default=None,
                        help="optional JSON to write rank and conditioning to")
    parser.add_argument("--quiet", action="store_true", help="suppress progress output")

    args = parser.parse_args(argv)
    create_covariance(args.corpus, output_path=args.output, form=args.form,
                      variance_keep=args.variance_keep, diagnostics_path=args.diagnostics,
                      show_progress=not args.quiet)


if __name__ == "__main__":
    main()
