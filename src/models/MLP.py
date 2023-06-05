# Internal Libraries
from .__IModel import IModel

# 3rd Party Libraries
import torch.nn as nn
import torch.nn.functional as F

class MLP(IModel):
    """
    Multilayer Perceptron (MLP) Modellklasse, die von der IModel Basisklasse erbt.

    Attributes:
        fc1 (nn.Linear): Erste Fully Connected Layer.
        fc2 (nn.Linear): Zweite Fully Connected Layer.
        fc3 (nn.Linear): Dritte Fully Connected Layer.
    """

    def __init__(self, hparams):
        """
        Initialisiert das MLP-Modell mit den gegebenen Hyperparametern.

        Args:
            hparams (dict): Enthält verschiedene Hyperparameter und Konfigurationsinformationen für das Modell.
        """
        super().__init__(hparams)

        # Definition der Fully Connected Layer
        self.fc1 = nn.Linear(23409, 500)
        self.fc2 = nn.Linear(500, 100)
        self.fc3 = nn.Linear(100, 6)

    def forward(self, x):
        """
        Definiert den Vorwärtsdurchgang für das MLP.

        Args:
            x (torch.Tensor): Eingabe-Datensatz.

        Returns:
            torch.Tensor: Ausgabe des Modells.
        """
        # Bestimmung der Batch-Größe
        batch_size, _, _, _ = x.size()

        # Ändern der Form der Eingabe auf ein eindimensionales Tensor
        x = x.view(batch_size, -1)

        # Anwendung der Fully Connected Layer und Verwendung der Leaky ReLU-Aktivierungsfunktion
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))

        # Anwendung der dritten Fully Connected Layer und Verwendung der LogSoftmax-Funktion
        x = F.log_softmax(self.fc3(x), dim=1)

        return x
