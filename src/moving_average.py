import pandas as pd
import sys
import yaml
from tqdm import tqdm
from dill import load, dump

# Collect command line arguments
stage_name = sys.argv[1]  # The stage name argument
input_filename = sys.argv[2]  # The input filename argument
output_filename = sys.argv[3]  # The output filename argument

# Load parameters from a YAML file
params = yaml.safe_load(open("params.yaml"))[stage_name]
moving_average_window_len_s = params["window_len_s"]  # Length of moving average window in seconds

# Print starting message
print(
    f"Calculating moving average for {input_filename} with window length {moving_average_window_len_s}s"
)

def moving_average(df: pd.DataFrame, window_len_s: float):
    """
    This function calculates the moving average for each column in the dataframe by grouping by hash.

    Args:
        df (pd.DataFrame): Dataframe with timestamps on index.
        window_len_s (float): Each window has this length in seconds.

    Returns:
        pd.DataFrame: Dataframe with the moving averages.
    """
    # Get frequency of timestamps
    freq = 1 / (df.index[1] - df.index[0]).total_seconds()

    # Determine number of observations per window
    window_size = int(window_len_s * freq)

    # Group dataframe by hash
    grouped = df.groupby("hash")

    # Select numeric columns only
    numeric_cols = df.select_dtypes(include=["number"]).columns

    # Calculate moving average for each group along the rows
    for name, group in grouped:
        df.loc[group.index, numeric_cols] = (
            group[numeric_cols].rolling(window_size, center=True, min_periods=1).mean()
        )

    return df

# Execute transformation
with open(input_filename, "rb") as fr:  # Load data
    data = {}

    # Iterate over each key-value pair in the loaded data
    for key, segments in tqdm(load(fr).items()):
        # Transform each segment using the moving average function
        data[key] = [
            moving_average(segment, moving_average_window_len_s) for segment in segments
        ]

    # Write transformed data to the output file
    with open(output_filename, "wb") as fw:
        dump(data, fw)
