# Internal Libraries
from .DataSetTabular import DataSetTabular
from .AbstractDataModule import AbstractDataModule

# 3rd Party Libraries
import pandas as pd
from dill import load
import torch

class DataModuleNDArray(AbstractDataModule):
    """
    DataModule for ndarray data with segment_id as the first component of the tuple (the other is the ndarray)
    """

    def get_dataset(self):
        # Load the predifined train test split with pandas
        train_test_split_ids = pd.read_json(
            self.train_test_split_filename, typ="series"
        )

        # Load the datasets
        with open(self.data_filename, "rb") as f:
            data = load(f)

        temp_train_data, temp_train_labels, temp_test_data, temp_test_labels = [], [], [], []
        for key, segments in data.items():
            # key consists of a tuple (activity, person, hash)
            # segments is a list of tuples (segment_id, segment)
            for segment_id, segment in segments:
                # segment is a 3d numpy array (columns, frequency components, time)
                if segment_id in train_test_split_ids["train"]:
                    temp_train_data.append(torch.from_numpy(segment))
                    temp_train_labels.append(self.onehotencode[key[0]])
                elif segment_id in train_test_split_ids["test"]:
                    temp_test_data.append(torch.from_numpy(segment))
                    temp_test_labels.append(self.onehotencode[key[0]])
                else:
                    raise ValueError(f"segment_id {segment_id} not in train or test")
                
        temp_test_data = torch.stack(temp_test_data)

        # Combine data and labels and convert to tensor
        train_data = DataSetTabular(temp_train_data, temp_train_labels)
        test_data = DataSetTabular(temp_test_data, temp_test_labels)

        return train_data, test_data
