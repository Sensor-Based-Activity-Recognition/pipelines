import sys
import pandas as pd
import yaml

from tqdm import tqdm
from dill import load

# Gather command line arguments
stage_name = sys.argv[1]  # Stage name argument
input_filename = sys.argv[2]  # Input filename argument
output_filename = sys.argv[3]  # Output filename argument

# Load parameters from a YAML file
params = yaml.safe_load(open("params.yaml"))[stage_name]
correlation_method = params["correlation_method"]  # Correlation method parameter

# Load data from a pickle file
with open(input_filename, "rb") as f:
    data = load(f)

warnings = []  # List to store warnings

def correlate(index, segment_data):
    """
    This function performs correlation between columns in a segment of data.

    Args:
        index (int): The index of the segment.
        segment_data (DataFrame): The segment data to be correlated.

    Returns:
        DataFrame: The dataframe containing the correlation results.
    """
    # Get segment id
    segment_id = segment_data["segment_id"][0]
    # Remove segment id from data
    segment_data = segment_data.drop(columns=["segment_id"])
    # Calculate correlations
    corr = segment_data.corr(numeric_only=True, method=correlation_method).stack()
    # Remove self correlations
    corr = corr[corr.index.get_level_values(0) != corr.index.get_level_values(1)]
    # Join index
    corr.index = corr.index.map("_corr_".join)
    # Convert to dataframe
    corr = pd.DataFrame(corr, columns=[index]).T

    # Join corr with activity, hash, person and segment id
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
        warnings.append((index, segment_id, all_same_value_cols))

    # Make sure we read cols and impute with 0 where correlation between two cols led to NA
    missing_columns = set(corr_column_names) - set(corr.columns)
    for col_left in missing_columns:
        corr[col_left] = 0
    return corr


# Create an empty dataframe for all correlations
corr_data = pd.DataFrame()

# Loop over measurements
for index_measurement, measurement in tqdm(data.items()):
    # Create an empty dataframe for correlations in the current measurement
    temp_segment = pd.DataFrame()

    # Loop over segments in the current measurement
    for segment in measurement:
        # Perform correlation for the current segment
        temp_corr = correlate(index_measurement, segment)

        # Add the current segment correlation to the measurement correlations
        temp_segment = pd.concat([temp_segment, temp_corr], axis=0)

    # Add the measurement correlations to the overall correlations
    corr_data = pd.concat([corr_data, temp_segment], axis=0)

# Print all warnings
affected_measurements = set()
for warning in warnings:
    affected_measurements.add(warning[0])
    print(f"Warning: Measurement {warning[0]} has segment {warning[1]} with cols {warning[2]} all same values. Correlation with other columns will be NA and therefore imputed to 0.")
print(f"Affected measurements: {affected_measurements}")

# Set 'segment_id' as index for the correlation data
corr_data = corr_data.set_index("segment_id")

# Save correlation data to a parquet file
corr_data.to_parquet(output_filename, index=True)
