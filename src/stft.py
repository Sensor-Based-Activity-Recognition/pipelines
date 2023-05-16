import sys
from dill import load, dump
import numpy as np
import pandas as pd
from scipy.signal import stft
from tqdm import tqdm

# get args
stage_name = sys.argv[1]
input_filename = sys.argv[2]
output_filename = sys.argv[3]

print(
    f"Short Time Fourier Transformation of {input_filename}"
)

def get_stft(df:pd.DataFrame):
    """get frequency spectrums from dataframe (column wise)

    Args:
        df (pd.DataFrame): dataframe with timestamps on index

    Returns:
        np.array: 3 dimensional array (columns, frequency components, time)
    """
    spectogram = []
    # for each numeric column
    for col in df.select_dtypes(include=np.number).columns:
        f, t, Zxx = stft(df[col], fs=50, noverlap=95, nperseg=100)
        spectogram.append(np.abs(Zxx))
    return np.array(spectogram)

# execute transformation
with open(input_filename, "rb") as fr: #load data
    data = {} #stft ified data stored here
    for key, segments in tqdm(load(fr).items()):
        data[key] = [get_stft(segment) for segment in segments]

    # save data
    with open(output_filename, "wb") as fw:
        dump(data, fw)
