from unittest import TestCase

import pytest
import torch

from pyrelational.model_managers.ensemble_model_manager import (
    LightningEnsembleModelManager,
)
from tests.test_utils import BreastCancerClassifier, get_classification_dataset


class TestEnsembleEstimator(TestCase):
    """Class containing unit tests for ensemble pyrelational model."""

    def setUp(self) -> None:
        """Set up shared attributes"""
        self.num_estimators = 4
        self.model = LightningEnsembleModelManager(
            BreastCancerClassifier, {}, {"epochs": 1}, n_estimators=self.num_estimators
        )
        self.dataset = get_classification_dataset()
        self.train_loader = self.dataset.get_train_loader()
        self.val_loader = self.dataset.get_validation_loader()

    def test_instantiation(self) -> None:
        """Check attributes at instantiation."""
        self.assertEqual(self.model.__class__.__name__, "LightningEnsembleModelManager")
        self.assertIsNone(self.model._current_model)
        self.assertIsInstance(self.model.trainer_config, dict)
        self.assertIsInstance(self.model.model_config, dict)

    def test_fail_on_test_without_train(self) -> None:
        """Check error is raised when testing without training first."""
        with pytest.raises(ValueError) as err:
            self.model.test(self.val_loader)
            self.assertEqual(
                str(err.value), "No current model, call 'train(train_loader, valid_loader)' to train the model first"
            )

    def test_prediction(self) -> None:
        """Check dimension match with number of estimators or dataset size."""
        self.model.train(self.train_loader)
        self.assertEqual(len(self.model._current_model), self.num_estimators)

        prediction = self.model(self.val_loader)
        self.assertEqual(prediction.size(0), self.num_estimators)
        self.assertEqual(prediction.size(1), len(self.dataset.validation_indices))
        self.assertIsInstance(prediction, torch.Tensor)
        self.assertIsInstance(self.model.test(self.val_loader), dict)
