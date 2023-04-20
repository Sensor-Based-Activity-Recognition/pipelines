import sys
import pandas as pd

# get args
input_filename = sys.argv[1]
feature_range_min = float(sys.argv[2])
feature_range_max = float(sys.argv[3])

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
data.to_parquet("data/min-max-scaled.parquet", index=True)
mins.to_csv("data/min-max-scaler-mins.csv", header=False)
maxs.to_csv("data/min-max-scaler-maxs.csv", header=False)
