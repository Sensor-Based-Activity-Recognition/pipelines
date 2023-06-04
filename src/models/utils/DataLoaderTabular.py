# Interne Bibliotheken
from .DataSetTabular import DataSetTabular
from .AbstractDataModule import AbstractDataModule

# 3rd-Party Bibliotheken
import pandas as pd
import torchvision.transforms as transforms
import torch
import numpy as np

class DataModuleTabular(AbstractDataModule):
    """
    DataModule für tabellarische Daten mit segment_id als Index.
    Dieses Modul lädt die Daten, teilt sie in Trainings- und Testdaten auf, führt Transformationen durch und bereitet sie für die weitere Verarbeitung vor.
    """

    def get_dataset(self):
        """
        Lädt die Daten, führt die Transformationen durch und teilt sie in Trainings- und Testdaten auf.

        Returns:
            train_data: Ein DataSetTabular-Objekt, das die Trainingsdaten enthält.
            test_data: Ein DataSetTabular-Objekt, das die Testdaten enthält.
        """

        # Lädt die vordefinierte Train-Test-Aufteilung mit pandas
        train_test_split_ids = pd.read_json(
            self.train_test_split_filename, typ="series"
        )

        # Lädt die Datasets
        data = pd.read_parquet(self.data_filename)

        # Teilt die Daten in Trainings- und Testdaten auf und konvertiert sie in numpy Arrays
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

        # Führt die Transformationen durch
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )
        temp_train_data = transform(temp_train_data)[0]
        temp_test_data = transform(temp_test_data)[0]

        # Kombiniert die Daten und die Labels und konvertiert sie in Tensoren
        train_data = DataSetTabular(temp_train_data, temp_train_labels)
        test_data = DataSetTabular(temp_test_data, temp_test_labels)

        return train_data, test_data
