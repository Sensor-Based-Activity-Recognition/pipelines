# Internal Libraries
from .__IModel import IModel

# 3rd Party Libraries
import torch.nn as nn
import torch.nn.functional as F

# ipython display
from IPython.display import display


class MLP(IModel):
    def __init__(self, hparams):
        super().__init__(hparams)
        # Fully connected layer 1
        self.fc1 = nn.Linear(23409, 500)
        # Fully connected layer 2
        self.fc2 = nn.Linear(500, 100)
        # Fully connected layer 3
        self.fc3 = nn.Linear(100, 6)

    def forward(self, x):
        batch_size, _, _, _ = x.size()
        # Flatten input tensor
        x = x.view(batch_size, -1)
        # Fully connected layer 1
        x = F.leaky_relu(self.fc1(x))
        # Fully connected layer 2
        x = F.leaky_relu(self.fc2(x))
        # Fully connected layer 3
        x = F.log_softmax(self.fc3(x), dim=1)
        return x
