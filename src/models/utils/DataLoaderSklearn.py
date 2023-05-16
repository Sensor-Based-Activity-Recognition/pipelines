import pandas as pd
import numpy as np
from dill import load

class DataLoaderSklearn_Tabular:
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

class DataLoaderSklearn_Segments:
    """
    DataModule for segmential data with segment_id index
    """

    def __init__(self, config, data_filename, train_test_split_filename):
        # Read all needed parameters
        self.train_test_split_filename = train_test_split_filename
        self.data_filename = data_filename
        self.config = config

        self.setup()

    def setup(self):
        # Load the predifined train test split with pandas
        train_test_split_ids = pd.read_json(
            self.train_test_split_filename, typ="series"
        )

        # Load the datasets
        with open(self.data_filename, "rb") as fr: #load data
            data:dict = load(fr)

        columns = self.config["columns"]

        #if columns is a string
        if type(columns) is str:
            if columns == "numeric":
                columns = list(data.values())[0][0].select_dtypes(include=["number"]).columns #get numeric columns
            else:
                raise Exception("unknown column identifier")

        train_ids = train_test_split_ids["train"]
        test_ids = train_test_split_ids["test"]

        x_train_temp = []
        y_train_temp = []
        x_test_temp = []
        y_test_temp = []

        for key, segments in data.items():
            activity = key[0]
            for segment in segments:
                segment_id = segment["segment_id"][0]

                features = segment[columns].to_numpy().transpose().flatten()
                y = onehotencode[activity]

                if segment_id in train_ids:
                    x_train_temp.append(features)
                    y_train_temp.append(y)

                elif segment_id in test_ids:
                    x_test_temp.append(features)
                    y_test_temp.append(y)

                else:
                    raise Exception(f"oh buoy nod gud... unknown segment id: {segment_id}")

        # Train data
        self.train_data = x_train_temp
        self.train_labels = y_train_temp

        # Test data
        self.test_data = x_test_temp
        self.test_labels = y_test_temp

# Load the OneHotEncodings
onehotencode = {
    "Sitzen": 0,
    "Laufen": 1,
    "Velofahren": 2,
    "Rennen": 3,
    "Stehen": 4,
    "Treppenlaufen": 5,
}