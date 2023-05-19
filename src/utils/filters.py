from argparse import ArgumentError
from scipy.signal import butter, lfilter
import pandas as pd
import numpy as np
from typing import Tuple, Union
import numbers


def butterworth_filter(
    df: pd.DataFrame,
    filter_type: str,
    cutoff_freq: Union[float, Tuple[float, float]],
    order: int = 3,
):
    """Apply filter on dataframe
    Args:
        df (pd.dataframe): dataframe
        filter_type (str): type of filter ('bandpass', 'lowpass', 'highpass')
        cutoff_freq (float | [float, float]): cutoff frequency(ies) for filter (for bandpass its: [lowcut, highcut])
        order (int): filter order
    Returns:
        dataframe with filtered columns (non numeric columns are deleted)
    """

    # assert constant frequency
    assert (np.diff(df.index.values) == np.diff(df.index.values)[0]).all()

    fs = pd.Timedelta(1, "s") / np.diff(df.index.values)[0]  # get sampling frequency
    nyq = 0.5 * fs  # calculate Nyquist frequency

    if filter_type == "bandpass":
        assert isinstance(cutoff_freq, tuple)
        btype = "band"
        cutoff = (cutoff_freq[0] / nyq, cutoff_freq[1] / nyq)

    elif filter_type == "lowpass":
        assert isinstance(cutoff_freq, numbers.Number)
        btype = "low"
        cutoff = cutoff_freq / nyq

    elif filter_type == "highpass":
        assert isinstance(cutoff_freq, numbers.Number)
        btype = "high"
        cutoff = cutoff_freq / nyq

    else:
        raise ArgumentError("unknown filter_type")

    b, a = butter(order, cutoff, fs=fs, btype=btype)  # create a corresponding filter

    # apply filter on columns
    data = df.copy()
    for col in df.select_dtypes(
        include=["number"]
    ).columns:  # only select numeric columns
        data[col] = lfilter(
            b, a, df[col].values
        )  # apply filter on column and store data

    return pd.DataFrame(
        data=data, index=df.index
    )  # return new dataframe with same index as original and filtered columns
