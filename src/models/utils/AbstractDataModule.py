# Importiert erforderliche Bibliotheken
from abc import ABC, abstractmethod
from torch import Generator
from torch.utils.data import random_split, DataLoader
from pytorch_lightning import LightningDataModule

class AbstractDataModule(LightningDataModule, ABC):
    """
    Abstrakte Klasse für ein LightningDataModule. Diese Klasse stellt gemeinsame Funktionen für
    LightningDataModule bereit und soll von spezifischen DataModules geerbt werden.

    Attribute:
        train_test_split_filename (str): Dateiname für den Train-Test-Split
        data_filename (str): Dateiname für die Daten
        seed (int): Seed-Wert für die Reproduzierbarkeit der Ergebnisse
        train_val_split (float): Verhältnis von Training zu Validierung
        batch_size (int): Größe der Datenbatches
        num_workers (int): Anzahl der Prozesse/Threads, die zum Laden der Daten verwendet werden sollen
    """
    def __init__(self, config, data_filename, train_test_split_filename):
        super().__init__()

        # Liest alle benötigten Parameter
        self.train_test_split_filename = train_test_split_filename
        self.data_filename = data_filename
        self.seed = config.seed
        self.train_val_split = config.data["train_val_split"]
        self.batch_size = config.data["batch_size"]
        self.num_workers = config.data["num_workers"]

    def prepare_data(self):
        """
        Wird verwendet, um Daten herunterzuladen. Hier jedoch nicht benötigt, daher pass.
        """
        pass

    @abstractmethod
    def get_dataset(self):
        """
        Gibt das Datenset zurück. 

        Returns:
            train_data: Das Training-Datenset
            test_data: Das Test-Datenset
        """
        pass

    def setup(self, stage="fit"):
        """
        Bereitet die Datensätze für das Training, die Validierung und das Testen vor.

        Args:
            stage (str, optional): Phase des Trainings ("fit", "validate" oder "test"). Defaults to "fit".
        """
        self.train_data, self.test_data = self.get_dataset()

        # Bühne für "fit" oder "validate" (Beide werden in einem Schritt generiert)
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

        # Bühne für "test" oder "predict" (Beide sind gleich)
        if stage == "test":
            self.test_dataset = self.test_data


    def train_dataloader(self):
        """
        Gibt den DataLoader für das Training zurück.

        Returns:
            DataLoader: DataLoader für das Training
        """
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            pin_memory=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        """
        Gibt den DataLoader für die Validierung zurück.

        Returns:
            DataLoader: DataLoader für die Validierung
        """
        return DataLoader(
            self.val_dataset,
            batch_size=len(self.val_dataset),
            pin_memory=True,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        """
        Gibt den DataLoader für das Testen zurück.

        Returns:
            DataLoader: DataLoader für das Testen
        """
        return DataLoader(
            self.test_dataset,
            batch_size=len(self.test_dataset),
            pin_memory=True,
            num_workers=self.num_workers
        )

    # Lädt die OneHot-Kodierungen
    onehotencode = {
        "Sitzen": 0,
        "Laufen": 1,
        "Velofahren": 2,
        "Rennen": 3,
        "Stehen": 4,
        "Treppenlaufen": 5,
    }
