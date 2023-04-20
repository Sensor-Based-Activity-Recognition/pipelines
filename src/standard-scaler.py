import sys
import pandas as pd
import yaml

# get args
stage_name = sys.argv[1]
input_filename = sys.argv[2]
output_filename_1 = sys.argv[3]
output_filename_2 = sys.argv[4]
output_filename_3 = sys.argv[5]

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
data.to_parquet(output_filename_1, index=True)
means.to_csv(output_filename_2, header=False)
stds.to_csv(output_filename_3, header=False)
