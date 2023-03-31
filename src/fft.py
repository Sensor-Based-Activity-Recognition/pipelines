import sys
from dill import load, dump
import pandas as pd
import numpy as np
from scipy.fft import fft, fftfreq

# get args
input_filename = sys.argv[1]
print(
    f"FFT transformation of {input_filename}"
)

def get_fft(df:pd.DataFrame):
    """get frequency spectrum from dataframe (column wise)

    Args:
        df (pd.DataFrame): dataframe with timestamps on index

    Returns:
        dataframe (index = frequency components, cols = corresponding fft)
    """
    
    #assert constant frequency
    assert (np.diff(df.index.values) == np.diff(df.index.values)[0]).all()

    fs = pd.Timedelta(1, "s") / np.diff(df.index.values)[0] #get sampling frequency

    N = len(df) #get length of dataframe
    xf = fftfreq(N, 1 / fs)[:N//2] #calculate fft sample frequencies

    #generate spectrum from each column
    data = {}
    for col in df.columns:
        if np.issubdtype(df[col].dtype, np.number): #only do fft transformation on numeric columns
            yf = fft(df[col].values)

            data[col] = 2.0/N * np.abs(yf[0:N//2])

    return pd.DataFrame(data=data, index=xf) #return dataframe with sample frequencies on index and corresponding fft on columns

# execute transformation
with open(input_filename, "rb") as fr: #load data
    data = {} #fft ified data stored here
    for key, segments in load(fr).items():
        data[key] = [get_fft(segment) for segment in segments]

    # dump fft of windows
    with open("data/fft.dill", "wb") as fw:
        dump(data, fw)
