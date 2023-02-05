from typing import Any, Optional, Tuple, Union

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from sklearn.datasets import load_breast_cancer, load_diabetes
from torch import Tensor
from torch.utils.data import Dataset

from pyrelational.data_managers.data_manager import DataManager


def get_regression_dataset(
    hit_ratio_at: Optional[Union[int, float]] = None,
    use_train: bool = True,
    use_validation: bool = True,
    use_test: bool = True,
    train_val_test_split: Tuple[int, int, int] = (350, 50, 42),
) -> DataManager:
    """
    Get datamanager which wraps diabetes regression dataset.

    :param hit_ratio_at: threshold for hit ratio
    :param use_train: whether to use provided train indices in datamanager
    :param use_validation: whether to use provided validation indices in datamanager
    :param use_test: whether to use provided test indices in datamanager
    :param train_val_test_split: tuple indicating the number of samples to put in train/val/test splits
    :return: pyrelational datamanager
    """
    pl.seed_everything(0)
    ds = DiabetesDataset()
    train_ds, valid_ds, test_ds = torch.utils.data.random_split(ds, train_val_test_split)
    train_indices = train_ds.indices
    valid_indices = valid_ds.indices
    test_indices = test_ds.indices
    return DataManager(
        ds,
        train_indices=train_indices if use_train else None,
        validation_indices=valid_indices if use_validation else None,
        test_indices=test_indices if use_test else None,
        loader_batch_size=10,
        hit_ratio_at=hit_ratio_at,
    )


def get_classification_dataset(
    labelled_size: Optional[int] = None,
    hit_ratio_at: Optional[Union[int, float]] = None,
    train_val_test_split: Tuple[int, int, int] = (400, 100, 69),
) -> DataManager:
    """
    Get datamanager which wraps sklearn breast cancer dataset.

    :param labelled_size: size of labelled set
    :param hit_ratio_at: set the top percentage threshold to compute hit ratio metric
    :param train_val_test_split: tuple indicating the number of samples to put in train/val/test splits
    :return: pyrelational datamanager
    """
    pl.seed_everything(0)

    ds = BreastCancerDataset()
    train_ds, valid_ds, test_ds = torch.utils.data.random_split(ds, train_val_test_split)
    train_indices = train_ds.indices
    valid_indices = valid_ds.indices
    test_indices = test_ds.indices
    labelled_indices = None if labelled_size is None else train_indices[:labelled_size]

    return DataManager(
        ds,
        train_indices=train_indices,
        validation_indices=valid_indices,
        test_indices=test_indices,
        labelled_indices=labelled_indices,
        loader_batch_size=10,
        hit_ratio_at=hit_ratio_at,
    )


class DiabetesDataset(Dataset):
    """A small regression dataset for examples"""

    def __init__(self):
        # Load the diabetes dataset
        diabetes_X, diabetes_y = load_diabetes(return_X_y=True)
        self.x = torch.FloatTensor(diabetes_X)
        self.y = torch.FloatTensor(diabetes_y)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class BreastCancerDataset(Dataset):
    """A small classification dataset for examples"""

    def __init__(self):
        super(BreastCancerDataset, self).__init__()
        sk_x, sk_y = load_breast_cancer(return_X_y=True)
        self.x = torch.FloatTensor(sk_x)
        self.y = torch.LongTensor(sk_y)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class BreastCancerClassifier(LightningModule):
    """Custom module for a simple classifier for breast cancer sklearn dataset"""

    def __init__(self, dropout_rate: float = 0, **kwargs: Any):
        super(BreastCancerClassifier, self).__init__()

        # input has 30 features and 2 classes
        self.layer_1 = nn.Linear(30, 16)
        self.layer_2 = nn.Linear(16, 8)
        self.dropout = nn.Dropout(dropout_rate)
        self.layer_3 = nn.Linear(8, 2)

    def forward(self, x: Tensor):
        x = self.layer_1(x)
        x = F.relu(x)
        x = self.layer_2(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.layer_3(x)

        x = F.log_softmax(x, dim=1)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        self.log("loss", loss.item())
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        self.log("test_loss", loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


class DiabetesRegressionModel(LightningModule):
    """Simple Regression model for diabetes dataset. It uses dropout for MCDropout."""

    def __init__(self, **kwargs: Any):
        super().__init__()
        self.layer_1 = nn.Linear(10, 8)
        self.layer_2 = nn.Linear(8, 4)
        self.layer_3 = nn.Linear(4, 1)
        self.elu = nn.ELU()
        self.softplus = nn.Softplus()
        self.dropout = nn.Dropout(0.3)
        self.criterion = nn.MSELoss()

    def forward(self, x: Tensor):
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
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4)
        return optimizer
