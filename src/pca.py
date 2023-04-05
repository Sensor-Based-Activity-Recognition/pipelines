import sys
import pandas as pd
from sklearn.decomposition import PCA

# get args
input_filename = sys.argv[1]
n_components = int(sys.argv[2])

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
pca_data.to_parquet("data/pca.parquet", index=True)
