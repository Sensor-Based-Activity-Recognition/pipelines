import sys
import pandas as pd
import yaml
from tqdm import tqdm

# get args
stage_name = sys.argv[1]
input_filename = sys.argv[2]
output_filename = sys.argv[3]

#get params
params = yaml.safe_load(open("params.yaml"))[stage_name]
resample_frequency_hz = params["resample_frequency_hz"]
interpolation_method = params["interpolation_method"]

print(
    f"Resampling {input_filename} to {resample_frequency_hz}Hz using {interpolation_method} interpolation"
)

# read parquet file
data = pd.read_parquet(input_filename)
data = data.set_index("timestamp", drop=True)
data = data.groupby("hash")

data_resampled = []

# resample by recording and combine in new dataframe
for _, recording in tqdm(data):
    recording = recording.resample(
        f"{int(1E6/resample_frequency_hz)}us", origin="start"
    ).interpolate(method=interpolation_method)

    # backward fill in case of missing values at start
    na_by_col = recording.isna().sum()
    for col in na_by_col:
        if col > 1:
            print(f"Warning: recording has more than 1 NA values in column with index {col}. Backward filling.")
    recording = recording.fillna(method="bfill")

    # save to combined dataframe
    data_resampled += [recording]

data_resampled = pd.concat(data_resampled)

data_resampled.to_parquet(output_filename, index=True)
