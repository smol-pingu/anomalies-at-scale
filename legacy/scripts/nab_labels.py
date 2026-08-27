import json

import numpy as np
import pandas as pd


def load_anomaly_timestamps(labels_path, relative_csv_path):
    """Load the ground-truth anomaly timestamps NAB records for one stream.

    Parameters
    ----------
    labels_path : str
        Path to a NAB labels JSON file (e.g. labels/combined_labels.json), mapping each
        stream's relative CSV path to a list of anomaly timestamp strings.
    relative_csv_path : str
        Key into that JSON, e.g. 'realAWSCloudwatch/ec2_cpu_utilization_ac20cd.csv'.

    Returns
    -------
    pd.DatetimeIndex
        The stream's labelled anomaly timestamps, empty if none are recorded.
    """
    with open(labels_path) as f:
        labels = json.load(f)
    return pd.DatetimeIndex(labels.get(relative_csv_path, []))


def label_windows(window_starts, window_length, anomaly_timestamps):
    """Label each window as anomalous if it contains a ground-truth-labelled anomaly.

    Parameters
    ----------
    window_starts : array-like of pd.Timestamp
        Start timestamp of each window.
    window_length : pd.Timedelta
        Duration spanned by each window, start to (inclusive) its last point.
    anomaly_timestamps : pd.DatetimeIndex
        Ground-truth anomaly timestamps for the stream the windows were drawn from, e.g.
        from `load_anomaly_timestamps`.

    Returns
    -------
    np.ndarray of str, shape (n_windows,)
        'anomalous' where the window's [start, start + window_length] span contains at
        least one of `anomaly_timestamps`, else 'normal'.
    """
    window_starts = pd.DatetimeIndex(window_starts)
    window_ends = window_starts + window_length

    if len(anomaly_timestamps) == 0:
        contains_anomaly = np.zeros(len(window_starts), dtype=bool)
    else:
        anomalies = pd.DatetimeIndex(anomaly_timestamps).to_numpy()
        starts = window_starts.to_numpy()
        ends = window_ends.to_numpy()
        contains_anomaly = (
            (anomalies[None, :] >= starts[:, None]) & (anomalies[None, :] <= ends[:, None])
        ).any(axis=1)

    return np.where(contains_anomaly, 'anomalous', 'normal')
