import sys
from dill import load, dump
import pandas as pd
import numpy as np
from scipy.fft import fft, fftfreq
from tqdm import tqdm

# get args
stage_name = sys.argv[1]
input_filename = sys.argv[2]
output_filename = sys.argv[3]

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
    for col in df.select_dtypes(include=["number"]).columns: #only select numeric columns
        yf = fft(df[col].values)

        data[col] = 2.0/N * np.abs(yf[0:N//2])
    
    df_transformed = pd.DataFrame(data=data, index=xf) #create dataframe a with sample frequencies on index and corresponding fft on columns

    #copy all non numerical values into dataframe
    for col in df.select_dtypes(exclude=["number"]).columns:
        series:np.array = df[col].values #copy column

        if len(np.unique(series)) != 1: #if series has more than one distinct value
            raise Exception("a non transformed column has more than one unique value... to prevent unexpected behavior, the transformation was stoped")
        
        df_transformed[col] = series[:len(xf)] #truncate values from column

        df_transformed[col] = df_transformed[col].astype(df[col].dtype) #set correct datatype

    return df_transformed #return dataframe 

# execute transformation
with open(input_filename, "rb") as fr: #load data
    data = {} #fft ified data stored here
    for key, segments in tqdm(load(fr).items()):
        data[key] = [get_fft(segment) for segment in segments]

    # dump fft of windows
    with open(output_filename, "wb") as fw:
        dump(data, fw)
