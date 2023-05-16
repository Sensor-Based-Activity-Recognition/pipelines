# Internal Libraries
from .DataSetTabular import DataSetTabular
from .AbstractDataModule import AbstractDataModule

# 3rd Party Libraries
import pandas as pd
import torchvision.transforms as transforms
import torch
import numpy as np
from torch import Generator
from torch.utils.data import random_split


class DataModuleTabular(AbstractDataModule):
    """
    DataModule for tabular data with segment_id index
    """

    def get_dataset(self):
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
        temp_train_labels = torch.from_numpy(
            np.vectorize(self.onehotencode.get)(
                data.loc[train_test_split_ids["train"]].iloc[:, -3].to_numpy()
            )
        )
        temp_test_data = (
            data.loc[train_test_split_ids["test"]]
            .iloc[:, :-3]
            .to_numpy(dtype=np.float32)
        )
        temp_test_labels = torch.from_numpy(
            np.vectorize(self.onehotencode.get)(
                data.loc[train_test_split_ids["test"]].iloc[:, -3].to_numpy()
            )
        )

        # Apply transforms
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )
        temp_train_data = transform(temp_train_data)[0]
        temp_test_data = transform(temp_test_data)[0]

        # Combine data and labels and convert to tensor
        train_data = DataSetTabular(temp_train_data, temp_train_labels)
        test_data = DataSetTabular(temp_test_data, temp_test_labels)

        return train_data, test_data
