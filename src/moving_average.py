import pandas as pd
import sys
import yaml
from tqdm import tqdm

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
    window_len = int(window_len_s * freq)
    # group by hash
    grouped = df.groupby("hash")
    numeric_cols = df.select_dtypes(include=["number"]).columns
    # calculate moving average for each group along the rows
    for name, group in tqdm(grouped):
        df.loc[group.index, numeric_cols] = group[numeric_cols].rolling(
            window_len
            ).mean()
    return df


if __name__ == "__main__":
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
    # read parquet file
    data = pd.read_parquet(input_filename)
    # calculate moving average
    data = moving_average(data, moving_average_window_len_s)
    # save dataframe
    data.to_parquet(output_filename)