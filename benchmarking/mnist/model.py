"""Model for MNIST classification.

Implementation is the same as BatchBald paper: https://arxiv.org/abs/1906.08158.
It can be found in their repository:
    https://github.com/BlackHC/BatchBALD/blob/master/src/mnist_model.py.
"""

from functools import partial
from typing import Tuple

import torch
from lightning.pytorch import LightningModule
from sklearn.metrics import accuracy_score
from torch import Tensor
from torch import nn as nn
from torch.nn import functional as F

from pyrelational.model_managers import LightningMCDropoutModelManager


class ConvNet(LightningModule):
    """Simple ConvNet for MNIST classification."""

    def __init__(self, num_classes: int):
        """Instantiate for ConvNet."""
        super().__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.conv1_drop = nn.Dropout2d()
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(1024, 128)
        self.fc1_drop = nn.Dropout()
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        """Run forward pass of the model."""
        x = F.relu(F.max_pool2d(self.conv1_drop(self.conv1(x)), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 1024)
        x = F.relu(self.fc1_drop(self.fc1(x)))
        x = self.fc2(x)
        x = F.log_softmax(x, dim=1)
        return x

    def training_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        """Run training step for the model."""
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        self.log("loss", loss)
        return loss

    def validation_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        """Run validation step for the model."""
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        return loss

    def test_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> None:
        """Run test step for the model."""
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        self.log("test_loss", loss)

        # compute accuracy
        _, y_pred = torch.max(logits.data, 1)
        accuracy = accuracy_score(y, y_pred)
        self.log("test_accuracy", accuracy)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Configure optimizer for the model."""
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer


MCConvNet = partial(
    LightningMCDropoutModelManager,
    model_class=ConvNet,
)
