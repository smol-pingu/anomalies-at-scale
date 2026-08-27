import itertools

import numpy as np
import pandas as pd
import iisignature
from tqdm import tqdm


def split_streams(df, time_col=0, reset_tol=0.0):
    """Split a long DataFrame of concatenated streams into a list of per-stream arrays.

    The time column increases within a stream (1, 2, 3, ...) then drops back down at the
    start of the next stream; a new stream is started wherever it fails to do so.

    Parameters
    ----------
    df : pd.DataFrame
        Long-format corpus: one row per time step, with the time column (`time_col`) first
        and every other column treated as a feature.
    time_col : str or int
        Name or positional index of the time column. Defaults to the first column.
    reset_tol : float
        A step is only treated as a reset once the time value drops by more than this
        tolerance, to absorb floating point noise in an otherwise-increasing time column.

    Returns
    -------
    list of np.ndarray
        One (length, 1 + n_features) array per stream, columns in the same order as `df`
        (time column first).
    """
    if len(df) == 0:
        return []

    if isinstance(time_col, int):
        time_col = df.columns[time_col]

    time = df[time_col].to_numpy(dtype=float)
    array = df.to_numpy(dtype=float)

    resets = np.where(np.diff(time) <= reset_tol)[0] + 1
    boundaries = np.concatenate(([0], resets, [len(time)]))

    return [array[boundaries[i]:boundaries[i + 1]] for i in range(len(boundaries) - 1)]


def signature_keys(width, trunc):
    """Word-keys for a signature up to level `trunc`, in iisignature's term order: level 1
    first, then level 2, ..., each level in lexicographic order over the alphabet 1..width."""
    keys = []
    for level in range(1, trunc + 1):
        keys.extend(itertools.product(range(1, width + 1), repeat=level))
    return keys


def _signature_columns(width, trunc):
    """Column names for a signature DataFrame: 'sig_()', 'sig_(1)', 'sig_(1,2)', ..."""
    columns = ['sig_()']
    columns += ['sig_({0})'.format(','.join(map(str, word))) for word in signature_keys(width, trunc)]
    return columns


def convert_np(df):
    """Stack the 'Sig' column of a signature DataFrame into a (n_streams, n_terms) array."""
    return np.vstack(df['Sig'].to_numpy())


def build_covariance_matrix(X):
    """Compute the pseudo-inverse covariance matrix used for Mahalanobis scoring."""
    mu = np.mean(X, axis=0)
    X_centered = X - mu
    U, s, Vt = np.linalg.svd(X_centered, full_matrices=False)
    covariance = np.dot(np.dot(Vt.T, np.diag(s**2)), Vt)
    covariance = covariance / X_centered.shape[0]
    return np.linalg.pinv(covariance, hermitian=True)


def transform_signatures(signatures, L_inv):
    """Transforms the signature corpus into a new basis to allow for L2 near-neighbor scoring to be equivalent to Mahalanobis scoring in the original basis."""
    signatures = np.asarray(signatures, dtype=float)
    if signatures.ndim == 1:
        signatures = signatures[None, :]
    return signatures @ L_inv.T


def vectorize(df, trunc=2, time_col=0, reset_tol=0.0, show_progress=False, chunk_size=1000,
              expand=True, keep_vector=True, output_path=None, L_inv=None,
              covariance_output_path=None):
    """Compute a DataFrame of signatures for a corpus of k-dimensional streams, along with
    the Mahalanobis covariance matrix fit on those (raw, unwhitened) signatures.

    Parameters
    ----------
    df : pd.DataFrame
        Long-format corpus in the layout described in `split_streams`: first column time
        step, remaining columns features, streams concatenated one after another.
    trunc : int
        Truncation level, i.e. the depth the signature is computed to.
    time_col : str or int
        Name or positional index of the time column. Defaults to the first column.
    reset_tol : float
        Passed to `split_streams`; see there.
    expand : bool
        Include one column per signature term.
    keep_vector : bool
        Include the raw signature vector in a 'Sig' column.
    output_path : str or None
        If given, also write the signature DataFrame to this path as parquet.
    L_inv : np.ndarray or None
        A pre-built Mahalanobis whitening matrix, as returned by `build_covariance_matrix`
        (e.g. the covariance matrix this same function returned for a training corpus). If
        given, `transform_signatures` is applied to the raw signatures with it before they're
        packaged into the returned DataFrame - useful for scoring new data against an
        existing corpus without re-fitting the covariance. The covariance matrix this call
        itself returns is always fit on the raw, unwhitened signatures, regardless of L_inv.
    covariance_output_path : str or None
        If given, also write the covariance matrix to this path as parquet, separate from
        `output_path`.

    Returns
    -------
    (pd.DataFrame, np.ndarray)
        The signature DataFrame - one row per stream, the signature vector in 'Sig' (with
        the leading constant 1.0 term, matching esig's convention; whitened by `L_inv` if one
        was given) and one column per signature term - and the Mahalanobis covariance matrix
        fit on this corpus's own raw signatures via `build_covariance_matrix`.
    """
    streams = split_streams(df, time_col=time_col, reset_tol=reset_tol)

    if not streams:
        df_vec = pd.DataFrame(columns=['Sig'])
        covariance_matrix = np.empty((0, 0))
    else:
        widths = {stream.shape[1] for stream in streams}
        if len(widths) > 1:
            raise ValueError('all streams must have the same number of feature columns, got {0}'.format(sorted(widths)))
        if any(stream.shape[0] < 2 for stream in streams):
            raise ValueError('every stream needs at least 2 points to compute a signature')

        width = streams[0].shape[1]

        # Compute the truncated signature of every stream, using iisignature. Streams may have
        # different lengths; within each chunk they are padded to the chunk's longest stream by
        # repeating the last point - a zero increment, so it leaves the signature unchanged -
        # which lets the whole chunk go through iisignature.sig in a single batched call.
        # iisignature.sig excludes the constant leading 1.0 term that esig includes; it is
        # prepended here to match esig's convention.
        sig_chunks = []
        chunks = [streams[i:i + chunk_size] for i in range(0, len(streams), chunk_size)]

        for chunk in tqdm(chunks, disable=not show_progress, desc='Computing signatures'):
            target_length = max(stream.shape[0] for stream in chunk)
            padded = [
                stream if stream.shape[0] == target_length
                else np.vstack((stream, np.repeat(stream[-1][np.newaxis, :], target_length - stream.shape[0], axis=0)))
                for stream in chunk
            ]
            batch_sigs = iisignature.sig(np.stack(padded), trunc)
            sig_chunks.append(np.concatenate([np.ones((len(chunk), 1)), batch_sigs], axis=1))

        sigs = np.concatenate(sig_chunks)

        covariance_matrix = build_covariance_matrix(sigs)

        if L_inv is not None:
            sigs = transform_signatures(sigs, L_inv)

        df_vec = pd.DataFrame(index=range(len(streams)))

        if keep_vector:
            df_vec['Sig'] = [sig for sig in sigs]

        if expand:
            columns = _signature_columns(width, trunc)
            df_vec = pd.concat([df_vec, pd.DataFrame(sigs, columns=columns, index=df_vec.index)], axis=1)

    if output_path is not None:
        df_vec.to_parquet(output_path, index=False)

    if covariance_output_path is not None:
        covariance_columns = [str(i) for i in range(covariance_matrix.shape[1])]
        pd.DataFrame(covariance_matrix, columns=covariance_columns).to_parquet(covariance_output_path, index=False)

    return df_vec, covariance_matrix
