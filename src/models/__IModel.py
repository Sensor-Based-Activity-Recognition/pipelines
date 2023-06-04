# 3rd Party Libraries
import torch.nn as nn
import torch.optim as optim
import torchmetrics as tm
from pytorch_lightning import LightningModule

class IModel(LightningModule):
    """
    Basisklasse für Modelle, die mit Pytorch Lightning trainiert werden sollen.

    Attributes:
        accuracy (tm.Accuracy): Accuracy-Metrik für Multiklassen-Aufgaben.
        f1 (tm.F1Score): F1-Metrik für Multiklassen-Aufgaben.
    """

    def __init__(self, hparams):
        """
        Initialisiert das Modell mit den gegebenen Hyperparametern.

        Args:
            hparams (dict): Enthält verschiedene Hyperparameter und Konfigurationsinformationen für das Modell.
        """
        super().__init__()
        self.save_hyperparameters(hparams)

        # Initialisierung der Metriken
        self.accuracy = tm.Accuracy(
            task="multiclass", num_classes=self.hparams.data["output_size"]
        )
        self.f1 = tm.F1Score(
            task="multiclass",
            num_classes=self.hparams.data["output_size"],
            average="macro",
        )

    def configure_optimizers(self):
        """
        Konfiguriert den Optimizer für das Modell.

        Returns:
            optim.SGD: Der konfigurierte SGD-Optimizer.
        """
        return optim.SGD(
            self.parameters(),
            lr=self.hparams.model_hparams["learning_rate"],
            weight_decay=self.hparams.model_hparams["weight_decay"],
            momentum=self.hparams.model_hparams["momentum"],
        )

    def training_step(self, batch, _):
        """
        Führt einen Trainingsschritt durch.

        Args:
            batch (Tuple[torch.Tensor]): Enthält die Daten und die zugehörigen Labels für diesen Batch.

        Returns:
            torch.Tensor: Der Verlust für diesen Trainingsschritt.
        """
        loss = self.__basic_step(batch, "train")
        self.log("train_loss", loss, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, _):
        """
        Führt einen Validierungsschritt durch.

        Args:
            batch (Tuple[torch.Tensor]): Enthält die Daten und die zugehörigen Labels für diesen Batch.
        """
        loss = self.__basic_step(batch, "val")
        self.log("val_loss", loss, on_step=True, on_epoch=True)

    def test_step(self, batch, _):
        """
        Führt einen Testschritt durch.

        Args:
            batch (Tuple[torch.Tensor]): Enthält die Daten und die zugehörigen Labels für diesen Batch.
        """
        self.__basic_step(batch, "test")

    # Private methods
    def __basic_step(self, batch, preset):
        """
        Führt einen Schritt durch, berechnet den Verlust und die Metriken und protokolliert sie.

        Args:
            batch (Tuple[torch.Tensor]): Enthält die Daten und die zugehörigen Labels für diesen Batch.
            preset (str): Gibt an, welcher Typ von Schritt (z.B. "train", "val", "test") durchgeführt wird.

        Returns:
            torch.Tensor: Der berechnete Verlust für diesen Schritt.
        """
        # Get the data and target
        data, target = batch
        output = self(data)

        # Calculate the loss and accuracy
        loss = nn.CrossEntropyLoss()(output, target)
        acc = self.accuracy(output, target)
        f1 = self.f1(output, target)

        # Logging
        self.log(f"{preset}_acc", acc, on_step=True, on_epoch=True)
        self.log(f"{preset}_f1", f1, on_step=True, on_epoch=True)

        return loss
