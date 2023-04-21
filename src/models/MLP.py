# Internal Libraries
from .__IModel import IModel

# 3rd Party Libraries
import torch.nn as nn
import torch.nn.functional as F


class MLP(IModel):
    def __init__(self, hparams):
        super().__init__(hparams)
        # Fully connected layer 1
        self.fc1 = nn.Linear(hparams.data["input_size"], 5 * hparams.data["input_size"])
        # Fully connected layer 2
        self.fc2 = nn.Linear(
            5 * hparams.data["input_size"], 25 * hparams.data["output_size"]
        )
        # Fully connected layer 3
        self.fc3 = nn.Linear(
            25 * hparams.data["output_size"], 25 * hparams.data["output_size"]
        )
        # Fully connected layer 4
        self.fc4 = nn.Linear(
            25 * hparams.data["output_size"], 5 * hparams.data["output_size"]
        )
        # Fully connected layer 5
        self.fc5 = nn.Linear(
            5 * hparams.data["output_size"], hparams.data["output_size"]
        )

    def forward(self, x):
        # Fully connected layer 1
        x = F.leaky_relu(self.fc1(x))
        # Fully connected layer 2
        x = F.leaky_relu(self.fc2(x))
        # Fully connected layer 3
        x = F.leaky_relu(self.fc3(x))
        # Fully connected layer 4
        x = F.leaky_relu(self.fc4(x))
        # Fully connected layer 5
        x = F.log_softmax(self.fc5(x), dim=1)
        return x
