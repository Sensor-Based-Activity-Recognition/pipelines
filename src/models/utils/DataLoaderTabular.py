# Internal Libraries
from .DataSetTabular import DataSetTabular

# 3rd Party Libraries
import pandas as pd
import torchvision.transforms as transforms
import torch
import numpy as np
from torch import Generator
from torch.utils.data import random_split, DataLoader
from pytorch_lightning import LightningDataModule


class DataModuleTabular(LightningDataModule):
    """
    DataModule for tabular data with segment_id index
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
        data = pd.read_parquet(self.data_filename)
        temp_train_data = data.iloc[train_test_split_ids["train"], :-3].to_numpy(
            dtype=np.float32
        )
        temp_test_data = data.iloc[train_test_split_ids["test"], :-3].to_numpy(
            dtype=np.float32
        )
        temp_train_labels = torch.from_numpy(
            np.vectorize(onehotencode.get)(
                data.iloc[train_test_split_ids["train"], -3].to_numpy()
            )
        )
        temp_test_labels = torch.from_numpy(
            np.vectorize(onehotencode.get)(
                data.iloc[train_test_split_ids["test"], -3].to_numpy()
            )
        )

        # Apply transforms
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
        )
        temp_train_data = transform(temp_train_data)[0]
        temp_test_data = transform(temp_test_data)[0]

        # Combine data and labels and convert to tensor
        train_data = DataSetTabular(temp_train_data, temp_train_labels)
        test_data = DataSetTabular(temp_test_data, temp_test_labels)

        # Stage fit or validate (Both are generated in one step)
        if stage in ("fit", "validate"):
            generator = Generator().manual_seed(self.seed)
            train_val_split = [
                self.train_val_split,
                1 - self.train_val_split,
            ]
            self.train_dataset, self.val_dataset = random_split(
                train_data,
                lengths=train_val_split,
                generator=generator,
            )

        # Stage test or predict (They are the same)
        if stage in ("test", "predict"):
            self.test_dataset = test_data
            self.predict_dataset = self.test_dataset

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def predict_dataloader(self):
        return self.test_dataloader()
