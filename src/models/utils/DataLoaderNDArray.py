# Importiert interne Bibliotheken
from .DataSetTabular import DataSetTabular
from .AbstractDataModule import AbstractDataModule

# Importiert externe Bibliotheken
import pandas as pd
from dill import load
import torch

class DataModuleNDArray(AbstractDataModule):
    """
    DataModule für ndarray-Daten mit segment_id als erstem Bestandteil des Tupels (der andere ist das ndarray).
    Erbt von der AbstractDataModule und implementiert die spezifische get_dataset Methode.
    """

    def get_dataset(self):
        """
        Lädt und bereitet das Dataset für das Training und das Testen vor.
        Die Funktion liest die vordefinierte Train-Test-Aufteilung und das eigentliche Dataset, 
        teilt sie entsprechend in Trainings- und Testdaten auf und konvertiert sie in PyTorch-Tensoren.

        Returns:
            train_data (DataSetTabular): Das Training-Datenset.
            test_data (DataSetTabular): Das Test-Datenset.
        """

        # Lädt die vordefinierte Train-Test-Aufteilung mit pandas
        train_test_split_ids = pd.read_json(
            self.train_test_split_filename, typ="series"
        )

        # Lädt die Datasets
        with open(self.data_filename, "rb") as f:
            data = load(f)

        # Temporäre Speicher für Daten und Labels
        temp_train_data, temp_train_labels, temp_test_data, temp_test_labels = [], [], [], []

        for key, segments in data.items():
            # Der Schlüssel besteht aus einem Tupel (Aktivität, Person, Hash)
            # Segmente ist eine Liste von Tupeln (segment_id, segment)
            for segment_id, segment in segments:
                # Das Segment ist ein 3D numpy array (Spalten, Frequenzkomponenten, Zeit)
                # Teilt die Daten in Trainings- und Testdaten auf der Grundlage der segment_id
                if segment_id in train_test_split_ids["train"]:
                    temp_train_data.append(torch.from_numpy(segment))
                    temp_train_labels.append(self.onehotencode[key[0]])
                elif segment_id in train_test_split_ids["test"]:
                    temp_test_data.append(torch.from_numpy(segment))
                    temp_test_labels.append(self.onehotencode[key[0]])
                else:
                    # Wenn die segment_id weder in den Trainings- noch in den Testdaten vorhanden ist, wirft einen Fehler
                    raise ValueError(f"segment_id {segment_id} not in train or test")
                
        # Stapelt alle Testdaten in einem Tensor
        temp_test_data = torch.stack(temp_test_data)

        # Kombiniert Daten und Labels und konvertiert sie in ein DataSetTabular-Objekt
        train_data = DataSetTabular(temp_train_data, temp_train_labels)
        test_data = DataSetTabular(temp_test_data, temp_test_labels)

        return train_data, test_data
