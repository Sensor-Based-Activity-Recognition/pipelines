# Internal Libraries
from .__IModel import IModel

# 3rd Party Libraries
import torch.nn as nn
import torch.nn.functional as F

class CNN(IModel):
    """
    Convolutional Neural Network (CNN) Modellklasse, die von der IModel Basisklasse erbt.

    Attributes:
        conv1 (nn.Conv2d): Erste Convolutional Layer.
        pool1 (nn.MaxPool2d): Erste Max Pooling Layer.
        conv2 (nn.Conv2d): Zweite Convolutional Layer.
        pool2 (nn.MaxPool2d): Zweite Max Pooling Layer.
        conv3 (nn.Conv2d): Dritte Convolutional Layer.
        gap (nn.AdaptiveAvgPool2d): Global Average Pooling Layer.
        fc1 (nn.Linear): Erste Fully Connected Layer.
        fc2 (nn.Sequential): Zweite Fully Connected Layer (Enthält eine Lineare Layer und eine LogSoftmax Layer).
    """

    def __init__(self, hparams):
        """
        Initialisiert das CNN-Modell mit den gegebenen Hyperparametern.

        Args:
            hparams (dict): Enthält verschiedene Hyperparameter und Konfigurationsinformationen für das Modell.
        """
        super().__init__(hparams)

        # Definition der Convolutional Layer und Max Pooling Layer
        self.conv1 = nn.Conv2d(in_channels=9, out_channels=32, kernel_size=5)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3)

        # Definition der Global Average Pooling Layer
        self.gap = nn.AdaptiveAvgPool2d(output_size=1)

        # Definition der Fully Connected Layer
        self.fc1 = nn.Linear(in_features=128, out_features=64)

        # Definition der LogSoftmax Layer
        self.fc2 = nn.Sequential(
            nn.Linear(in_features=64, out_features=hparams.data["output_size"]),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        """
        Definiert den Vorwärtsdurchgang für das CNN.

        Args:
            x (torch.Tensor): Eingabe-Datensatz.

        Returns:
            torch.Tensor: Ausgabe des Modells.
        """
        # Anwendung der Convolutional Layer und Pooling Layer
        x = F.leaky_relu(self.conv1(x))
        x = self.pool1(x)
        x = F.leaky_relu(self.conv2(x))
        x = self.pool2(x)
        x = F.leaky_relu(self.conv3(x))

        # Anwendung der Global Average Pooling Layer
        x = self.gap(x)

        # Ändern der Form der Eingabe, um sie durch die Fully Connected Layer zu füttern
        x = x.view(-1, 128)

        # Anwendung der Fully Connected Layer
        x = F.leaky_relu(self.fc1(x))
        x = self.fc2(x)

        return x
