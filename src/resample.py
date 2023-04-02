import sys
import pandas as pd

# get args
input_filename = sys.argv[1]
resample_frequency_hz = int(sys.argv[2])
interpolation_method = sys.argv[3]

print(
    f"Resampling {input_filename} to {resample_frequency_hz}Hz using {interpolation_method} interpolation"
)

# read parquet file
data = pd.read_parquet(input_filename)
data = data.set_index("timestamp", drop=True)
data = data.groupby("hash")

data_resampled = []

# resample by recording and combine in new dataframe
for _, recording in data:
    recording = recording.resample(
        f"{int(1E6/resample_frequency_hz)}us", origin="start"
    ).interpolate(method=interpolation_method)

    # backward fill in case of missing values at start
    recording = recording.fillna(method="bfill")

    # save to combined dataframe
    data_resampled += [recording]

data_resampled = pd.concat(data_resampled)

data_resampled.to_parquet("data/resample.parquet", index=True)
