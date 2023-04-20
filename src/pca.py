import sys
import pandas as pd
from sklearn.decomposition import PCA
import yaml

# get args
stage_name = sys.argv[1]
input_filename = sys.argv[2]
output_filename = sys.argv[3]

#get params
params = yaml.safe_load(open("params.yaml"))[stage_name]
n_components = params["n_components"]

# read parquet file
data = pd.read_parquet(input_filename)

# reset index
data = data.reset_index(drop=False)

# get numeric and non numeric columns
numeric_data = data._get_numeric_data().columns
non_numeric_data = data.columns.difference(numeric_data)

# set non numeric data as index
data = data.set_index(non_numeric_data.to_list())

# pca on numeric data
pca = PCA(n_components=n_components)
pca_data = pca.fit_transform(data)

# convert to dataframe
pca_data = pd.DataFrame(
    pca_data, columns=[f"pca_{i+1}" for i in range(n_components)], index=data.index
)

# convert index to columns
pca_data = pca_data.reset_index(drop=False)

# if there is a column named "timestamp", set it as index
if "timestamp" in pca_data.columns:
    pca_data = pca_data.set_index("timestamp")

# save to parquet
pca_data.to_parquet(output_filename, index=True)
