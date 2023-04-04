from dill import dump
import pandas as pd
import sys

# get args
input_filename = sys.argv[1]
window_len_s = float(sys.argv[2])
overlap_percent = int(sys.argv[3])

print(
    f"Segmentating {input_filename} to windows of {window_len_s}s with {overlap_percent}% overlap"
)

# read parquet file
data = pd.read_parquet(input_filename)

# add segment id column
data.loc[:, "segment_id"] = -1
segment_id = 0

# helper function
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
    Remark: last window (|11,12,13|) wouldn't have full length which is why this data is ignored
    Returns:
        list of dataframes
    """
    overlap_timedelta = pd.Timedelta((window_len_s / 100) * overlap_percent, "s")  

    windows = []
    window_start = df.index[0]
    while(True):
        window_end = window_start + pd.Timedelta(window_len_s, "s")

        # window cannot reach full length anymore
        if window_end > df.index[-1]:
            return windows

        selected_rows = (df.index >= window_start) & (df.index <= window_end)

        # add segment id
        global segment_id
        df.loc[selected_rows, "segment_id"] = segment_id
        segment_id += 1

        windows.append(df.loc[selected_rows])
        window_start = window_end - overlap_timedelta

# segmentate
window_dict = {}
for group_name, dataframe in data.groupby(["activity", "person", "hash"]):
    window_dict[group_name] = segmentate(dataframe, window_len_s, overlap_percent)

# dump windows
with open("data/segmentate.dill", "wb") as f:
    dump(window_dict, f)
