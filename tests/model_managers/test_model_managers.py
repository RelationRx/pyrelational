from typing import Tuple
from unittest import TestCase

import torch
from torch.utils.data import DataLoader

from pyrelational.model_managers import (
    LightningEnsembleModelManager,
    LightningMCDropoutModelManager,
    LightningModelManager,
)
from tests.test_utils import DiabetesDataset, DiabetesRegressionModel


class TestModelManager(TestCase):
    """Class containing unit tests for pyrelational models."""

    def test_lightning_model(self) -> None:
        """
        Check that
        1) model is stored after training
        2) output of test loop is a dictionary
        3) shape of tensor output of __call__
        """
        train_loader, valid_loader, test_loader = get_loaders()
        model = LightningModelManager(DiabetesRegressionModel, {}, {"epochs": 3})
        model.train(train_loader, valid_loader)
        self.assertIsNotNone(model._current_model)
        self.assertIsInstance(model.test(test_loader), dict)
        self.assertEqual(model(test_loader).size(0), len(test_loader.dataset))

    def test_early_stopping_in_trainer_callbacks(self) -> None:
        """Check that EarlyStopping is one of the callbacks in a pyrelational LightningModelManager."""
        model = LightningModelManager(
            DiabetesRegressionModel, {}, {"epochs": 3, "use_early_stopping": True, "patience": 10}
        )
        trainer, _ = model.init_trainer()
        self.assertTrue(any(["EarlyStopping" in str(cb) for cb in trainer.callbacks]))


def get_loaders() -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train/val/test dataloaders from sklearn diabetes dataset."""
    ds = DiabetesDataset()
    train_ds, valid_ds, test_ds = torch.utils.data.random_split(ds, [350, 50, 42])

    train_loader = DataLoader(train_ds, batch_size=10)
    valid_loader = DataLoader(valid_ds, batch_size=10)
    test_loader = DataLoader(test_ds, batch_size=10)
    return train_loader, valid_loader, test_loader
