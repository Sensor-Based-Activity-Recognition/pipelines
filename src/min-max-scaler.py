import sys
import pandas as pd
import yaml

# get args
stage_name = sys.argv[1]
input_filename = sys.argv[2]
output_filename_1 = sys.argv[3]
output_filename_2 = sys.argv[4]
output_filename_3 = sys.argv[5]

# get config
params = yaml.safe_load(open("params.yaml"))[stage_name]
feature_range_min = params["feature_range_min"]
feature_range_max = params["feature_range_max"]

# read parquet file
data = pd.read_parquet(input_filename)

# get only numeric columns
numeric_data = data._get_numeric_data()

# get mins
mins = numeric_data.min()
# get maxs
maxs = numeric_data.max()

# min-max scale
data[numeric_data.columns] = (numeric_data - mins) / (maxs - mins)
# scale to feature range
data[numeric_data.columns] = (
    data[numeric_data.columns] * (feature_range_max - feature_range_min)
    + feature_range_min
)

# save scaled data, mins and maxs
data.to_parquet(output_filename_1, index=True)
mins.to_csv(output_filename_2, header=False)
maxs.to_csv(output_filename_3, header=False)
