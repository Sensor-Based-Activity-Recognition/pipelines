from scipy.signal import butter, lfilter
import pandas as pd
import numpy as np
from dill import load, dump
import sys

input_filename = sys.argv[1]
lowcut = float(sys.argv[2])
highcut = float(sys.argv[3])
order = int(sys.argv[4])
print(
    f"Apply bandpass filter on {input_filename} with low cutoff of {lowcut}Hz and high cutoff of {highcut}Hz and order of {order}"
)

def butt_bandpass(df:pd.DataFrame, lowcut_f:float, highcut_f:float, order:int=3):
    """ Apply bandpass (let trough frequencies between lowcut_f and highcut_f) filter on dataframe
    Args:
        df (pd.dataframe): dataframe
        lowcut_f (float): low cutoff frequency
        highcut_f (float): high cutoff frequency
        order (int): filter order
    Returns:
        dataframe with filtered columns (non numeric columns are deleted)
    """
    
    #assert constant frequency
    assert (np.diff(df.index.values) == np.diff(df.index.values)[0]).all()

    fs = pd.Timedelta(1, "s") / np.diff(df.index.values)[0] #get sampling frequency
    nyq = 0.5 * fs #calculate Nyquist frequency

    b, a = butter(order, [lowcut_f/nyq, highcut_f/nyq], fs=fs, btype='band') #create a bandpass filter

    #apply filter on columns
    data = df.copy()
    for col in df.columns:
        if np.issubdtype(df[col].dtype, np.number): #only do filtering on numeric columns
            data[col] = lfilter(b, a, df[col].values) #apply filter on column and store data

    return pd.DataFrame(data=data, index=df.index) #return new dataframe with same index as original and filtered columns

# execute filtering
with open(input_filename, "rb") as fr: #load data
    data = {} #filtered data stored he§re
    for key, segments in load(fr).items():
        data[key] = [butt_bandpass(segment, lowcut, highcut, order) for segment in segments]

    # dump fft of windows
    with open("data/bandpass.dill", "wb") as fw:
        dump(data, fw)