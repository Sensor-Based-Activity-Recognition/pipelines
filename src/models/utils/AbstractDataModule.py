from abc import ABC, abstractmethod
from torch import Generator
from torch.utils.data import random_split, DataLoader
from pytorch_lightning import LightningDataModule

class AbstractDataModule(LightningDataModule, ABC):
    """
    Abstract DataModule for common functionality
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

    @abstractmethod
    def get_dataset(self):
        """
        Returns the dataset

        Returns:
            train_data: The training dataset
            test_data: The test dataset
        """
        pass

    def setup(self, stage="fit"):
        self.train_data, self.test_data = self.get_dataset()

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
            batch_size=self.batch_size,
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


    # Load the OneHotEncodings
    onehotencode = {
        "Sitzen": 0,
        "Laufen": 1,
        "Velofahren": 2,
        "Rennen": 3,
        "Stehen": 4,
        "Treppenlaufen": 5,
    }
