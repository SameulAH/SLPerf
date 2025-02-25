#!/usr/bin/python3

"""models.py Contains an implementation of the LeNet5 model

For the ID2223 Scalable Machine Learning course at KTH Royal Institute of
Technology"""

__author__ = "Xenia Ioannidou and Bas Straathof"

import torch
import torch.nn as nn
import torch.nn.functional as F


class LeNetClientNetworkPart1(nn.Module):
    """CNN following the architecture of:
    https://medium.com/ml2vec/intro-to-pytorch-with-image-classification-on-a- \
            fashion-clothes-dataset-e589682df0c5

    The ClientNetwork is used for Split Learning and implements the CNN
    until the first convolutional layer."""

    def __init__(self):
        super(LeNetClientNetworkPart1, self).__init__()

        # First block - convolutional
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

    def forward(self, x):
        """Defines forward pass of CNN until the split layer, which is the first
        convolutional layer

        Args:
            x: Input Tensor

        Returns:
          x: Output Tensor
        """
        # Apply first convolutional block to input tensor
        x = self.block1(x)

        return x


class LeNetClientNetworkPart2(nn.Module):
    """CNN following the architecture of:
    https://medium.com/ml2vec/intro-to-pytorch-with-image-classification-on-a- \
            fashion-clothes-dataset-e589682df0c5

    The ServerNetwork is used for Split Learning and implements the CNN
    from the split layer until the last."""

    def __init__(self):
        super(LeNetClientNetworkPart2, self).__init__()

        # Third block - fully connected
        self.block3 = nn.Sequential(
            nn.Linear(in_features=256, out_features=120),
            nn.ReLU(),
            nn.Linear(in_features=120, out_features=84),
            nn.ReLU(),
            nn.Linear(in_features=84, out_features=10))

    def forward(self, x):
        """Defines forward pass of CNN from the split layer until the last

        Args:
            x: Input Tensor

        Returns:
          x: Output Tensor
        """
        # Apply fully-connected block to input tensor
        x = self.block3(x)
        return x

####### Ismail's code #######

class LeNetClientNetworkPart11(nn.Module):
    """Client-side Part 1 of the U-shaped network.
    Updated to accept CIFAR-10 RGB images (3 channels).
    """
    def __init__(self):
        super(LeNetClientNetworkPart11, self).__init__()
        self.block1 = nn.Sequential(
            # Updated in_channels from 1 to 3 for CIFAR-10
            nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        return self.block1(x)


class LeNetClientNetworkPart21(nn.Module):
    """Client-side Part 2 of the U-shaped network.
    Processes the flattened output from the server-side network.
    
    Note: The output of LeNetServerNetwork_U is 16 channels with a spatial
    dimension of 5x5 (for CIFAR-10), which gives 16*5*5 = 400 features.
    Thus, we update the first linear layer to accept 400 features.
    """
    def __init__(self):
        super(LeNetClientNetworkPart21, self).__init__()
        self.block3 = nn.Sequential(
            nn.Linear(in_features=400, out_features=120),  # Updated from 256 to 400
            nn.ReLU(),
            nn.Linear(in_features=120, out_features=84),
            nn.ReLU(),
            nn.Linear(in_features=84, out_features=10)
        )

    def forward(self, x):
        return self.block3(x)



class LeNetServerNetwork_U1(nn.Module):
    """Server-side network in the U-shaped split learning architecture.
    Processes the activations from the client part 1.
    """
    def __init__(self):
        super(LeNetServerNetwork_U1, self).__init__()
        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        x = self.block2(x)
        # Flatten the output for the fully connected layers in client part 2.
        x = x.view(x.size(0), -1)
        return x



##############################












class LeNetServerNetwork_U(nn.Module):
    """CNN following the architecture of:
    https://medium.com/ml2vec/intro-to-pytorch-with-image-classification-on-a- \
            fashion-clothes-dataset-e589682df0c5

    The ServerNetwork is used for Split Learning and implements the CNN
    from the split layer until the last."""

    def __init__(self):
        super(LeNetServerNetwork_U, self).__init__()

        # Second block - convolutional
        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

    def forward(self, x):
        """Defines forward pass of CNN from the split layer until the last

        Args:
            x: Input Tensor

        Returns:
          x: Output Tensor
        """
        # Apply second convolutional block to input tensor
        x = self.block2(x)

        # Flatten output
        # x = x.view(-1, 4*4*16)
        x = x.view(x.size(0), -1)

        return x




class adult_LR_U_client1(nn.Module):
    def __init__(self):
        super(adult_LR_U_client1, self).__init__()
        # First block - convolutional
        self.block1 = nn.Sequential(
            nn.Linear(24, 10),

        )

    def forward(self, x):
        x = self.block1(x)

        return x


class adult_LR_U_server(nn.Module):
    def __init__(self):
        super(adult_LR_U_server, self).__init__()
        # First block - convolutional

        self.block2 = nn.Sequential(
            nn.Linear(10, 7),

        )

    def forward(self, x):
        x = self.block2(x)
        return x


class adult_LR_U_client2(nn.Module):
    def __init__(self):
        super(adult_LR_U_client2, self).__init__()
        # First block - convolutional
        self.block3 = nn.Sequential(
            nn.Linear(7, 2),

        )

    def forward(self, x):
        x = self.block3(x)
        return x
