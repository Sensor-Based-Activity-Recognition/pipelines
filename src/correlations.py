import sys
import pandas as pd
import yaml

from tqdm import tqdm
from dill import load

# get args
stage_name = sys.argv[1]
input_filename = sys.argv[2]
output_filename = sys.argv[3]

# get params
params = yaml.safe_load(open("params.yaml"))[stage_name]
correlation_method = params["correlation_method"]

# read pickle file
with open(input_filename, "rb") as f:
    data = load(f)

warnings = []

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

    # join corr with activity, hash, person and segment id
    corr["activity"] = segment_data["activity"][0]
    corr["hash"] = segment_data["hash"][0]
    corr["person"] = segment_data["person"][0]
    corr["segment_id"] = segment_id

    # Generate correlation column names so we can check existence of all columns later
    corr_column_names = []
    segment_data_features = segment_data.select_dtypes(include="float32")
    for col_left in segment_data_features.columns:
        for col_right in segment_data_features.columns:
            if col_left != col_right:
                corr_column_names.append(f"{col_left}_corr_{col_right}")

    # Warn if segment has columns with all same values
    all_same_value_cols = []
    for col in segment_data_features.columns:
        if segment_data_features[col].nunique() == 1:
            all_same_value_cols.append(col)
    if len(all_same_value_cols) > 0:
        # append to warnings dict with index as key
        warnings.append((index, segment_id, all_same_value_cols))

    # Make sure we read cols and impute with 0 where correlation between two cols led to NA
    missing_columns = set(corr_column_names) - set(corr.columns)
    for col_left in missing_columns:
        corr[col_left] = 0
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

# print warnings
affected_measurements = set()
for warning in warnings:
    affected_measurements.add(warning[0])
    print(f"Warning: Measurement {warning[0]} has segment {warning[1]} with cols {warning[2]} all same values. Correlation with other columns will be NA and therefore imputed to 0.")
print(f"Affected measurements: {affected_measurements}")

# set index
corr_data = corr_data.set_index("segment_id")

# save to parquet
corr_data.to_parquet(output_filename, index=True)
