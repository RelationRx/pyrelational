import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from sklearn.metrics import accuracy_score


class DiabetesRegression(LightningModule):
    """Simple Regression model for diabetes dataset

    It uses dropout for MCDropout to be used
    """

    def __init__(self, dropout=0):
        super(DiabetesRegression, self).__init__()
        self.layer_1 = nn.Linear(10, 8)
        self.layer_2 = nn.Linear(8, 4)
        self.layer_3 = nn.Linear(4, 1)
        self.elu = nn.ELU()
        self.softplus = nn.Softplus()
        self.dropout = nn.Dropout(dropout)
        self.criterion = nn.MSELoss()

    def forward(self, x):
        x = self.dropout(self.elu(self.layer_1(x)))
        x = self.dropout(self.elu(self.layer_2(x)))
        x = self.dropout(self.layer_3(x))
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        predictions = self(x)
        loss = self.criterion(predictions, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        predictions = self(x)
        loss = self.criterion(predictions, y)
        self.log("loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        predictions = self(x)
        loss = self.criterion(predictions, y)
        self.log("test_loss", loss)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4)
        return optimizer


class BreastCancerClassification(LightningModule):
    """Simple classification model for cancer dataset"""

    def __init__(self, dropout=0):
        super(BreastCancerClassification, self).__init__()
        self.layer_1 = nn.Linear(30, 16)
        self.layer_2 = nn.Linear(16, 8)
        self.layer_3 = nn.Linear(8, 2)
        self.elu = nn.ELU()
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=1)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        x = self.dropout(self.elu(self.layer_1(x)))
        x = self.dropout(self.elu(self.layer_2(x)))
        x = self.dropout(self.layer_3(x))
        x = self.softmax(x)
        return x

    def return_penultimate_embed(self, x: torch.Tensor):
        """
        Return embedding from penultimate layer.

        :param x: input tensor to calculate embedding for
        """
        x = self.dropout(self.elu(self.layer_1(x)))
        x = self.dropout(self.elu(self.layer_2(x)))
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        self.log("loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        self.log("test_loss", loss)
        # return loss

        # compute accuracy
        _, y_pred = torch.max(logits.data, 1)
        accuracy = accuracy_score(y, y_pred)
        self.log("accuracy", accuracy)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4)
        return optimizer


class MnistClassification(LightningModule):
    """Custom module for a simple convnet Classification"""

    def __init__(self, dropout=0):
        super(MnistClassification, self).__init__()

        # mnist images are (1, 28, 28) (channels, width, height)
        self.layer_1 = nn.Linear(28 * 28, 128)
        self.layer_2 = nn.Linear(128, 256)
        self.layer_3 = nn.Linear(256, 10)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        batch_size, channels, width, height = x.size()

        # (b, 1, 28, 28) -> (b, 1*28*28)
        x = x.view(batch_size, -1)
        x = self.dropout(self.layer_1(x))
        x = F.relu(x)
        x = self.dropout(self.layer_2(x))
        x = F.relu(x)
        x = self.dropout(self.layer_3(x))

        x = F.log_softmax(x, dim=1)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        self.log("loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        self.log("test_loss", loss)

        # compute accuracy
        _, y_pred = torch.max(logits.data, 1)
        accuracy = accuracy_score(y, y_pred)
        self.log("test_accuracy", accuracy)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
