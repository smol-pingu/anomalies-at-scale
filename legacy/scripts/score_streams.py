import numpy as np
import faiss

from signature_computer import convert_np

def faiss_kNN(transformed_corpus, transformed_anomalies, k=6):
    """Finds the k nearest neighbors of each anomaly in the transformed corpus using FAISS, and returns the distances and indices of the nearest neighbors.

    transformed_corpus, transformed_anomalies : pd.DataFrame
        Signature DataFrames as returned by signature_computer.vectorize (a 'Sig' column of
        signature vectors, one row per stream).
    """
    corpus = convert_np(transformed_corpus).astype(np.float32)
    anomalies = convert_np(transformed_anomalies).astype(np.float32)

    # Create a FAISS index for the corpus
    d = corpus.shape[1]  # Dimensionality of the data
    index = faiss.IndexFlatL2(d)  # L2 distance index
    index.add(corpus)  # Add corpus to the index

    # Search for the k nearest neighbors of each anomaly
    distances, indices = index.search(anomalies, k)

    return distances, indices

def exact_kNN(transformed_corpus, transformed_test, k=6):
    """Finds the k nearest neighbors of each test signature in the transformed corpus using
    exact (brute-force) Euclidean distance, and returns the distances and indices of the
    nearest neighbors.

    Parameters
    ----------
    transformed_corpus : pd.DataFrame
        Signature DataFrame for the reference corpus, as returned by signature_computer.vectorize
        (a 'Sig' column of signature vectors, one row per stream).
    transformed_test : pd.DataFrame
        Signature DataFrame for the new/test data, in the same format as transformed_corpus.
    k : int
        Number of nearest neighbors to return for each test signature.

    Returns
    -------
    distances : np.ndarray, shape (n_test, k)
        Euclidean distance from each test signature to its k nearest corpus signatures,
        sorted ascending.
    indices : np.ndarray, shape (n_test, k)
        Row index into transformed_corpus of each of those k nearest neighbors.
    """
    corpus = convert_np(transformed_corpus)
    test = convert_np(transformed_test)

    if corpus.shape[1] != test.shape[1]:
        raise ValueError('corpus and test signatures must have the same dimension, got {0} and {1}'.format(
            corpus.shape[1], test.shape[1]))

    k = min(k, corpus.shape[0])

    # Compute pairwise distances between every test signature and every corpus signature.
    distances_full = np.linalg.norm(test[:, np.newaxis, :] - corpus[np.newaxis, :, :], axis=2)

    # Partition out the k smallest per row, then sort just those k into ascending order.
    nearest = np.argpartition(distances_full, k - 1, axis=1)[:, :k]
    row_idx = np.arange(distances_full.shape[0])[:, np.newaxis]
    nearest_distances = distances_full[row_idx, nearest]

    order = np.argsort(nearest_distances, axis=1)
    indices = nearest[row_idx, order]
    distances = nearest_distances[row_idx, order]

    return distances, indices

def score_streams(transformed_corpus, transformed_test, k=6, use_faiss=True):
    """Scores each test signature by the average distance to its k nearest neighbors in the
    transformed corpus.

    Parameters
    ----------
    transformed_corpus : pd.DataFrame
        Signature DataFrame for the reference corpus, as returned by signature_computer.vectorize
        (a 'Sig' column of signature vectors, one row per stream).
    transformed_test : pd.DataFrame
        Signature DataFrame for the new/test data, in the same format as transformed_corpus.
    k : int
        Number of nearest neighbors to use for scoring.
    use_faiss : bool
        If True, use FAISS for approximate nearest neighbor search; otherwise, use exact search.

    Returns
    -------
    scores : np.ndarray, shape (n_test,)
        Average distance from each test signature to its k nearest corpus signatures.
    """
    Threshold = 1e-10 #To be built from calibration set
    if use_faiss:
        distances, _ = faiss_kNN(transformed_corpus, transformed_test, k=k)
    else:   
        distances, _ = exact_kNN(transformed_corpus, transformed_test, k=k)

    scores = np.mean(distances, axis=1)
    scores_max = np.max(distances, axis=1)
    scores_min = np.min(distances, axis=1)
    return scores, scores_max, scores_min
    #Provides the average distance to the k nearest neighbors as the score, and the maximum and minimum distances to the k nearest neighbors as additional metrics.
def Calibration(transformed_corpus, transformed_test, k=6, use_faiss=True):
    """Calibrates the threshold for anomaly detection based on the scores of the test signatures.

    Parameters
    ----------
    transformed_corpus : pd.DataFrame
        Signature DataFrame for the reference corpus, as returned by signature_computer.vectorize
        (a 'Sig' column of signature vectors, one row per stream).
    transformed_test : pd.DataFrame
        Signature DataFrame for the new/test data, in the same format as transformed_corpus.
    k : int
        Number of nearest neighbors to use for scoring.
    use_faiss : bool
        If True, use FAISS for approximate nearest neighbor search; otherwise, use exact search.

    Returns
    -------
    Threshold : float
        The calibrated threshold value for classifying anomalies.
    """
    scores, _, _ = score_streams(transformed_corpus, transformed_test, k=k, use_faiss=use_faiss)
    Threshold = np.percentile(scores, 95)  # Set threshold at the 95th percentile of scores
    return Threshold

def Thresholding(scores, Threshold):
    """Applies a threshold to the scores to classify anomalies.

    Parameters
    ----------
    scores : np.ndarray, shape (n_test,)
        Average distance from each test signature to its k nearest corpus signatures.
    Threshold : float
        The threshold value for classifying anomalies.

    Returns
    -------
    anomaly_flags : np.ndarray, shape (n_test,)
        Boolean array indicating whether each test signature is classified as an anomaly (True) or not (False).
    """
    anomaly_flags = scores > Threshold
    return anomaly_flags