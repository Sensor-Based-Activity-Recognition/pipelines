# 3rd Party Libraries
import torch.nn as nn
import torch.optim as optim
import torchmetrics as tm
import torch.nn.functional as F
from pytorch_lightning import LightningModule


class IModel(LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)

        # Metrics
        self.accuracy = tm.Accuracy(
            task="multiclass", num_classes=self.hparams.data["output_size"]
        )
        self.f1 = tm.F1Score(
            task="multiclass",
            num_classes=self.hparams.data["output_size"],
            average="macro",
        )

    def loss_fn(self, output, target):
        target = F.one_hot(target, num_classes=self.hparams.data["output_size"]).float()
        return nn.CrossEntropyLoss()(output, target)

    def configure_optimizers(self):
        return optim.SGD(
            self.parameters(),
            lr=self.hparams.model_hparams["learning_rate"],
            weight_decay=self.hparams.model_hparams["weight_decay"],
            momentum=self.hparams.model_hparams["momentum"],
        )

    def training_step(self, batch, _):
        return self.__basic_step(batch, "val")

    def validation_step(self, batch, _):
        return self.__basic_step(batch, "val")

    def test_step(self, batch, _):
        return self.__basic_step(batch, "test")

    # Private methods
    def __basic_step(self, batch, preset):
        # Get the data and target
        data, target = batch
        output = self(data)

        # Calculate the loss and accuracy
        loss = self.loss_fn(output, target)
        acc = self.accuracy(output, target)
        f1 = self.f1(output, target)

        # Logging
        self.log(f"{preset}_loss", loss, on_step=True, on_epoch=True)
        self.log(f"{preset}_acc", acc, on_step=True, on_epoch=True)
        self.log(f"{preset}_f1", f1, on_step=True, on_epoch=True)

        return loss
