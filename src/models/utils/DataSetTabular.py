from torch.utils.data import Dataset

class DataSetTabular(Dataset):
    """
    Eine benutzerdefinierte Dataset-Klasse für tabellarische Daten.
    Es implementiert die notwendigen Methoden (__init__, __len__ und __getitem__), um mit PyTorch DataLoader genutzt werden zu können.

    Attributes:
        data (torch.Tensor): Die Eingangsdaten.
        labels (torch.Tensor): Die zugehörigen Labels zu den Eingangsdaten.
    """

    def __init__(
        self,
        data,
        labels,
    ):
        """
        Initialisiert das Dataset mit Daten und Labels.

        Args:
            data (torch.Tensor): Die Eingangsdaten.
            labels (torch.Tensor): Die zugehörigen Labels zu den Eingangsdaten.
        """
        self.data = data
        self.labels = labels

    def __len__(self):
        """
        Gibt die Länge des Datasets zurück.

        Returns:
            int: Die Anzahl der Datenpunkte im Dataset.
        """
        return len(self.labels)

    def __getitem__(self, idx):
        """
        Ermöglicht die Indexierung des Datasets, so dass es möglich ist, auf ein bestimmtes Datum und Label zuzugreifen.

        Args:
            idx (int): Der Index des gewünschten Datenpunkts.

        Returns:
            Tuple[torch.Tensor]: Ein Tupel, bei dem das erste Element das Datum und das zweite Element das zugehörige Label ist.
        """
        return self.data[idx], self.labels[idx]
