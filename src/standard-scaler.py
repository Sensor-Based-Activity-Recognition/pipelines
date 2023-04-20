import sys
import pandas as pd

# get args
input_filename = sys.argv[1]

# read parquet file
data = pd.read_parquet(input_filename)

# get only numeric columns
numeric_data = data._get_numeric_data()

# get means
means = numeric_data.mean()
# get stds
stds = numeric_data.std()

# standardize
data[numeric_data.columns] = (numeric_data - means) / stds

# save standardized data, means and stds
data.to_parquet("data/standard-scaled.parquet", index=True)
means.to_csv("data/standard-scaler-means.csv", header=False)
stds.to_csv("data/standard-scaler-stds.csv", header=False)
