# Internal Libraries

# 3rd Party Libraries
import pandas as pd
import numpy as np

class DataLoaderSklearn:
    """
    DataModule for tabular data with segment_id index
    """

    def __init__(self, config, data_filename, train_test_split_filename):
        # Read all needed parameters
        self.train_test_split_filename = train_test_split_filename
        self.data_filename = data_filename

        self.setup()

    def setup(self):
        # Load the predifined train test split with pandas
        train_test_split_ids = pd.read_json(
            self.train_test_split_filename, typ="series"
        )

        # Load the OneHotEncodings
        onehotencode = {
            "Sitzen": 0,
            "Laufen": 1,
            "Velofahren": 2,
            "Rennen": 3,
            "Stehen": 4,
            "Treppenlaufen": 5,
        }

        # Load the datasets
        data = pd.read_parquet(self.data_filename)
        temp_train_data = (
            data.loc[train_test_split_ids["train"]]
            .iloc[:, :-3]
            .to_numpy(dtype=np.float32)
        )
        temp_train_labels = np.vectorize(onehotencode.get)(
            data.loc[train_test_split_ids["train"]].iloc[:, -3].to_numpy()
        )
        temp_test_data = (
            data.loc[train_test_split_ids["test"]]
            .iloc[:, :-3]
            .to_numpy(dtype=np.float32)
        )
        temp_test_labels = np.vectorize(onehotencode.get)(
            data.loc[train_test_split_ids["test"]].iloc[:, -3].to_numpy()
        )

        # Train data
        self.train_data = temp_train_data
        self.train_labels = temp_train_labels

        # Test data
        self.test_data = temp_test_data
        self.test_labels = temp_test_labels
