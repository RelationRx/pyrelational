import pytorch_lightning as pl
import torch
import torch.nn as nn
from sklearn.datasets import load_diabetes
from torch.utils.data import DataLoader, Dataset

from pyrelational.models import (
    LightningEnsembleModel,
    LightningMCDropoutModel,
    LightningModel,
)


def test_lightning_model():
    train_loader, valid_loader, test_loader = get_loaders()
    model_class = get_model()
    model = LightningModel(model_class, {}, {"epochs": 3})
    model.train(train_loader, valid_loader)
    assert model.current_model is not None
    assert isinstance(model.test(test_loader), dict)
    assert model(test_loader).size(0) == len(test_loader.dataset)

    model = LightningModel(model_class, {}, {"epochs": 3, "use_early_stopping": True, "patience": 10})
    trainer, _ = model.init_trainer()
    assert any(["EarlyStopping" in str(cb) for cb in trainer.callbacks])


def test_ensemble_model():
    train_loader, valid_loader, test_loader = get_loaders()
    model_class = get_model()
    model = LightningEnsembleModel(model_class, {}, {"epochs": 3}, n_estimators=3)
    assert len(model.init_model()) == 3
    model.train(train_loader, valid_loader)
    assert model(test_loader).size(0) == 3
    assert len(model.current_model) == 3


def test_mcdropout_model():
    train_loader, valid_loader, test_loader = get_loaders()
    model_class = get_model()
    model = LightningMCDropoutModel(model_class, {}, {"epochs": 3}, n_estimators=3)
    model.train(train_loader, valid_loader)
    assert model(test_loader).size(0) == 3


def get_model():
    class DiabetesRegression(pl.LightningModule):
        """Simple Regression model for diabetes dataset

        It uses dropout for MCDropout to be used
        """

        def __init__(self, **kwargs):
            super(DiabetesRegression, self).__init__()
            self.layer_1 = nn.Linear(10, 8)
            self.layer_2 = nn.Linear(8, 4)
            self.layer_3 = nn.Linear(4, 1)
            self.elu = nn.ELU()
            self.softplus = nn.Softplus()
            self.dropout = nn.Dropout(0.3)
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
            return loss

        def configure_optimizers(self):
            optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4)
            return optimizer

    return DiabetesRegression


def get_loaders():
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

    ds = DiabetesDataset()
    train_ds, valid_ds, test_ds = torch.utils.data.random_split(ds, [350, 50, 42])

    train_loader = DataLoader(train_ds, batch_size=10)
    valid_loader = DataLoader(valid_ds, batch_size=10)
    test_loader = DataLoader(test_ds, batch_size=10)
    return train_loader, valid_loader, test_loader
