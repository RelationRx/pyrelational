from unittest import TestCase

import pytest
import torch

from pyrelational.model_managers.mcdropout_model_manager import (
    LightningMCDropoutModelManager,
    _check_mc_dropout_model,
    _enable_only_dropout_layers,
)
from tests.test_utils import BreastCancerClassifier, get_classification_dataset


class TestMCDropoutModel(TestCase):
    """Class containing unit tests for MCDropoutModel and associated utility functions."""

    def test_enable_only_dropout_layers(self) -> None:
        """Check that _enable_only_dropout_layers enables dropout layers in eval mode."""
        model = BreastCancerClassifier(dropout_rate=0.5)
        model.eval()
        _enable_only_dropout_layers(model, p=0.25)
        self.assertTrue(model.dropout.training)
        self.assertEqual(model.dropout.p, 0.25)
        self.assertTrue(model.layer_1.training == model.layer_2.training == model.layer_3.training is False)

    def test_check_mc_dropout_model_pass(self) -> None:
        """Verify that _check_mc_dropout_model passes when provided with a model with dropout modules."""
        _check_mc_dropout_model(BreastCancerClassifier, {"dropout_rate": 0.5})

    def test_check_mc_dropout_model_raise_error(self) -> None:
        """Verify that _check_mc_dropout_model raises an error when provided with a model without dropout modules."""

        class Model(torch.nn.Module):
            def __init__(self):
                super(Model, self).__init__()

        with pytest.raises(ValueError) as err:
            _check_mc_dropout_model(Model, {})
        self.assertEqual(
            str(err.value),
            "Model provided do not contain any torch.nn.Dropout modules, cannot apply MC Dropout",
        )

    def test_model_names(self) -> None:
        """Check model manager class name and model class name."""
        model = LightningMCDropoutModelManager(
            BreastCancerClassifier,
            {"dropout_rate": 0.5},
            {"epochs": 1},
            eval_dropout_prob=0.2,
            n_estimators=3,
        )
        self.assertEqual(model.__class__.__name__, "LightningMCDropoutModelManager")
        self.assertEqual(model._init_model().__class__.__name__, "BreastCancerClassifier")

    def test_mc_dropout_estimator(self) -> None:
        """Check model prediction shape."""
        dataset = get_classification_dataset()
        train_loader = dataset.get_train_loader()
        val_loader = dataset.get_validation_loader()
        model = LightningMCDropoutModelManager(
            BreastCancerClassifier,
            {"dropout_rate": 0.5},
            {"epochs": 1},
            eval_dropout_prob=0.2,
            n_estimators=3,
        )
        model.train(train_loader)
        prediction = model(val_loader)
        self.assertEqual(prediction.size(0), 3)
        self.assertEqual(prediction.size(1), len(dataset.validation_indices))
        self.assertIsInstance(prediction, torch.Tensor)
