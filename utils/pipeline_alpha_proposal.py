import pandas as pd
import numpy as np
from scipy.fft import fft, fftfreq
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.decomposition import PCA

def resample_df(df:pd.DataFrame, fs_resample_Hz:int, interp_method:"str"="linear"):
    """Resample (interpolate) dataframe to a certain frequency
    Args:
        df (pd.DataFrame): dataframe with timestamps on index
    
    Returns:
        resampled pd.DataFrame
    """
    return df.resample(f"{int(1E6/fs_resample_Hz)}us", origin="start").interpolate(method=interp_method)

def get_fft_freqency_spectrum(df:pd.DataFrame):
    """get frequency spectrum from dataframe (column wise)

    Args:
        df (pd.DataFrame): dataframe with timestamps on index

    Returns:
        dict(key: column, value: absolute fft)
    """
    
    #assert constant frequency
    assert (np.diff(df.index.values) == np.diff(df.index.values)[0]).all()

    fs = pd.Timedelta(1, "s") / np.diff(df.index.values)[0] #get sampling frequency

    #generate spectrum from each column
    data = {}
    for col in df.columns:
        N = len(df[col])

        yf = fft(df[col].values)
        xf = fftfreq(N, 1 / fs)

        data[col] = (xf[:N//2], 2.0/N * np.abs(yf[0:N//2]))

    return data

def segmentate(df:pd.DataFrame, window_len_s:float, overlap_percent:int):
    """Makes windows [aka best os ;)] from dataframe
    Args:
        df (pd.DataFrame): dataframe with timestamps on index
        window_len_s (float): each window has this length in seconds
        overlap_percent (int): percentage of overlap from previous window [0, 100]

    Example window length 4s and 50% overlap:
    
    1. original time series (each number represents a second): |1,2,3,4,5,6,7,8,9,10,11,12,13|
    
    2. use function: segmentate(|1,2,3,4,5,6,7,8,9,10,11,12,13|, 4, 50)

    3. returns: [|1,2,3,4|,|3,4,5,6|,|5,6,7,8|,|7,8,9,10|,|9,10,11,12|]

    Remark: last window (|11,12,13|) wouldn't have full length why this data is ignored

    Returns:
        list of dataframes
    """
    
    overap_timedelta = pd.Timedelta((window_len_s / 100) * overlap_percent, "s")  

    windows = []
    window_start = df.index[0]
    while(True):
        window_end = window_start + pd.Timedelta(window_len_s, "s")

        #window cannot reach full length
        if window_end > df.index[-1]:
            return windows

        windows.append(df.loc[(df.index >= window_start) & (df.index <= window_end)])
        
        window_start = window_end - overap_timedelta


#TODO proposed pipelines

#1 (this pipeline completely ignores a person during train)
#raw data -> train test split (leave one person out of train) -> train -> segmentate -> fft -> merge/shuffle -> pca -> train classifier -> classifier
#                                                             |                                                  |                         |
#                                                             -> test  -> segmentate -> fft -> merge/shuffle -> pca transform -----------> test classifier -> result                       

#2 (this pipeline tests on a person which was also trained on)
#raw data -> segmentate -> fft train test split -> merge/shuffle -> pca -> train test split -> train -> train classifier -> classifier
#                                                                                           |                               |
#                                                                                           -> test  -> ------------------> test classifier -> result