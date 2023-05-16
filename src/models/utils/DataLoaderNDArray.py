# Internal Libraries
from .DataSetTabular import DataSetTabular

# 3rd Party Libraries
import pandas as pd
from dill import load
import torchvision.transforms as transforms
import torch
import numpy as np
from torch import Generator
from torch.utils.data import random_split, DataLoader
from pytorch_lightning import LightningDataModule


class DataModuleNDArray(LightningDataModule):
    """
    DataModule for ndarray data with segment_id as the first component of the tuple (the other is the ndarray)
    """

    def __init__(self, config, data_filename, train_test_split_filename):
        super().__init__()

        # Read all needed parameters
        self.train_test_split_filename = train_test_split_filename
        self.data_filename = data_filename
        self.seed = config.seed
        self.train_val_split = config.data["train_val_split"]
        self.batch_size = config.data["batch_size"]
        self.num_workers = config.data["num_workers"]

    def prepare_data(self):
        ## Used for downloading data, we don't need it
        pass

    def setup(self, stage="fit"):
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
                    temp_train_labels.append(onehotencode[key[0]])
                elif segment_id in train_test_split_ids["test"]:
                    temp_test_data.append(torch.from_numpy(segment))
                    temp_test_labels.append(onehotencode[key[0]])
                else:
                    raise ValueError(f"segment_id {segment_id} not in train or test")
                
        temp_test_data = torch.stack(temp_test_data)
       
        # Combine data and labels and convert to tensor
        self.train_data = DataSetTabular(temp_train_data, temp_train_labels)
        self.test_data = DataSetTabular(temp_test_data, temp_test_labels)

        # Stage fit or validate (Both are generated in one step)
        if stage in ("fit", "validate"):
            generator = Generator().manual_seed(self.seed)
            train_val_split = [
                self.train_val_split,
                1 - self.train_val_split,
            ]
            self.train_dataset, self.val_dataset = random_split(
                self.train_data,
                lengths=train_val_split,
                generator=generator,
            )

        # Stage test or predict (They are the same)
        if stage == "test":
            self.test_dataset = self.test_data

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            pin_memory=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=len(self.val_dataset),
            pin_memory=True,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=len(self.test_dataset),
            pin_memory=True,
            num_workers=self.num_workers
        )
