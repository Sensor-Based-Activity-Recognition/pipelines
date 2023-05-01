import sys
import pandas as pd
import yaml

from tqdm import tqdm
from dill import load

# get args
stage_name = sys.argv[1]
input_filename = sys.argv[2]
output_filename = sys.argv[3]

#get params
params = yaml.safe_load(open("params.yaml"))[stage_name]
correlation_method = params["correlation_method"]

# read pickle file
with open(input_filename, "rb") as f:
    data = load(f)


def correlate(index, segment_data):
    """
    Correlate columns in segments.
    """
    # get segment id
    segment_id = segment_data["segment_id"][0]
    # remove segment id from data
    segment_data = segment_data.drop(columns=["segment_id"])
    # get correlations
    corr = segment_data.corr(numeric_only=True, method=correlation_method).stack()
    # remove self correlations
    corr = corr[corr.index.get_level_values(0) != corr.index.get_level_values(1)]
    # join index
    corr.index = corr.index.map("_corr_".join)
    # convert to dataframe
    corr = pd.DataFrame(corr, columns=[index]).T
    # join corr with activity, hash, person and segment id,
    corr["activity"] = segment_data["activity"][0]
    corr["hash"] = segment_data["hash"][0]
    corr["person"] = segment_data["person"][0]
    corr["segment_id"] = segment_id
    # return data
    return corr


# create empty dataframe for all correlations
corr_data = pd.DataFrame()
# loop over measurements
for index_measurement, measurement in tqdm(data.items()):
    # create empty dataframe for correlations in measurement
    temp_segment = pd.DataFrame()
    # loop over segments
    for segment in measurement:
        # correlate columns in segment
        temp_corr = correlate(index_measurement, segment)
        # add to segment correlations
        temp_segment = pd.concat([temp_segment, temp_corr], axis=0)
    # add to all correlations
    corr_data = pd.concat([corr_data, temp_segment], axis=0)
# set index
corr_data = corr_data.set_index("segment_id")

# drop rows with NA values
na_by_col = corr_data.isna().sum()
for n in na_by_col:
    if n > 1:
        print(f"Warning: corr_data has at least {n} unexpected NA values. Dropping affected rows.")
        # corr_data = corr_data.dropna()
        break

# TODO: remove, bfill NA for testing purpose
# corr_data = corr_data.fillna(method="bfill")

# save to parquet
corr_data.to_parquet(output_filename, index=True)
