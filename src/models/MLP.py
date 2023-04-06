# Internal Libraries
from .__IModel import IModel

# 3rd Party Libraries
import torch.nn as nn
import torch.nn.functional as F


class MLP(IModel):
    def __init__(self, hparams):
        super().__init__(hparams)
        # Fully connected layer 1
        self.fc1 = nn.Linear(hparams.input_size, 5 * hparams.input_size)
        self.fc1_bn = nn.BatchNorm1d(5 * hparams.input_size)
        # Fully connected layer 2
        self.fc2 = nn.Linear(5 * hparams.input_size, 25 * hparams.output_size)
        self.fc2_bn = nn.BatchNorm1d(25 * hparams.output_size)
        # Fully connected layer 3
        self.fc3 = nn.Linear(25 * hparams.output_size, 25 * hparams.output_size)
        self.fc3_bn = nn.BatchNorm1d(25 * hparams.output_size)
        # Fully connected layer 4
        self.fc4 = nn.Linear(25 * hparams.output_size, 5 * hparams.output_size)
        self.fc4_bn = nn.BatchNorm1d(5 * hparams.output_size)
        # Fully connected layer 5
        self.fc5 = nn.Linear(5 * hparams.output_size, hparams.output_size)
        self.fc5_bn = nn.BatchNorm1d(hparams.output_size)

    def forward(self, x):
        # Fully connected layer 1
        x = F.leaky_relu(self.fc1_bn(self.fc1(x)))
        # Fully connected layer 2
        x = F.leaky_relu(self.fc2_bn(self.fc2(x)))
        # Fully connected layer 3
        x = F.leaky_relu(self.fc3_bn(self.fc3(x)))
        # Fully connected layer 4
        x = F.leaky_relu(self.fc4_bn(self.fc4(x)))
        # Fully connected layer 5
        x = F.log_softmax(self.fc5_bn(self.fc5(x)), dim=1)
        return x
