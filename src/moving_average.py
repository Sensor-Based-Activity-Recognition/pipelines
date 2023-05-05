import pandas as pd
import sys
import yaml
from tqdm import tqdm
from dill import load, dump

# get args
stage_name = sys.argv[1]
input_filename = sys.argv[2]
output_filename = sys.argv[3]

#get params
params = yaml.safe_load(open("params.yaml"))[stage_name]
moving_average_window_len_s = params["window_len_s"]

print(
    f"Calculating moving average for {input_filename} with window length {moving_average_window_len_s}s"
)

# helper function
def moving_average(df:pd.DataFrame, window_len_s:float):
    """Calculates moving average for each column in dataframe by grouping by hash
    Args:
        df (pd.DataFrame): dataframe with timestamps on index
        window_len_s (float): each window has this length in seconds
    """

    # get frequency of timestamps
    freq = 1/(df.index[1] - df.index[0]).total_seconds()

    # get number of observations per window
    window_size = int(window_len_s * freq)

    # group by hash
    grouped = df.groupby("hash")
    numeric_cols = df.select_dtypes(include=["number"]).columns

    # calculate moving average for each group along the rows
    for name, group in grouped:
        df.loc[group.index, numeric_cols] = group[numeric_cols].rolling(window_size).mean()

    return df

# execute transformation
with open(input_filename, "rb") as fr: #load data
    data = {} #fft ified data stored here
    for key, segments in tqdm(load(fr).items()):
        data[key] = [moving_average(segment, moving_average_window_len_s) for segment in segments]

    # dump fft of windows
    with open(output_filename, "wb") as fw:
        dump(data, fw)