from scipy.signal import butter, lfilter
import pandas as pd
import numpy as np
from dill import load, dump
import sys

input_filename = sys.argv[1]
cutoff = float(sys.argv[2])
order = int(sys.argv[3])
print(
    f"Apply lowpass filter on {input_filename} with cutoff of {cutoff}Hz and order of {order}"
)

def butt_lowpass(df:pd.DataFrame, cut_f:float, order=3):
    """ Apply lowpass (let trough frequencies below cut_f) filter on dataframe
    Args:
        df (pd.dataframe): dataframe
        cut_f (float): cut off frequency
        order (int): filter order
    Returns:
        dataframe with filtered columns (non numeric columns are deleted)
    """
    
    #assert constant frequency
    assert (np.diff(df.index.values) == np.diff(df.index.values)[0]).all()

    fs = pd.Timedelta(1, "s") / np.diff(df.index.values)[0] #get sampling frequency
    nyq = 0.5 * fs #calculate Nyquist frequency

    b, a = butter(order, cut_f/nyq, fs=fs, btype='low') #create a lowpass filter

    #apply filter on columns
    data = df.copy()
    for col in df.columns:
        if np.issubdtype(df[col].dtype, np.number): #only do filtering on numeric columns
            data[col] = lfilter(b, a, df[col].values) #apply filter on column and store data

    return pd.DataFrame(data=data, index=df.index) #return new dataframe with same index as original and filtered columns

# execute filtering
with open(input_filename, "rb") as fr: #load data
    data = {} #filtered data stored here
    for key, segments in load(fr).items():
        data[key] = [butt_lowpass(segment, cutoff, order) for segment in segments]

    # dump fft of windows
    with open("data/lowpass.dill", "wb") as fw:
        dump(data, fw)