# Internal Libraries
from .__IModel import IModel

# 3rd Party Libraries
import torch.nn as nn
import torch.nn.functional as F

class CNN(IModel):
    def __init__(self, hparams):
        super().__init__(hparams)

        # Convolutional layer 1
        self.conv1 = nn.Conv2d(
            in_channels=9,
            out_channels=32,
            kernel_size=5
        )
        self.pool1 = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )
        # Convolutional layer 2
        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=4
        )
        self.pool2 = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )
        # Convolutional layer 3
        self.conv3 = nn.Conv2d(
            in_channels=64,
            out_channels=128,
            kernel_size=3
        )
        # global average pooling layer
        self.gap = nn.AdaptiveAvgPool2d(
            output_size=1
        )
        # Fully connected layer 1
        self.fc1 = nn.Linear(
            in_features=128,
            out_features=64
        )
        # softmax layer
        self.fc2 = nn.Sequential(
            nn.Linear(
                in_features=64,
                out_features=hparams.data["output_size"]
            ),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x))
        x = self.pool1(x)
        x = F.leaky_relu(self.conv2(x))
        x = self.pool2(x)
        x = F.leaky_relu(self.conv3(x))
        x = self.gap(x)
        x = x.view(-1, 128)
        x = F.leaky_relu(self.fc1(x))
        x = self.fc2(x)
        return x


