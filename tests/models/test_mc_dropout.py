import os
import sys

import pytest
import torch

from pyrelational.models.mcdropout_model import (
    LightningMCDropoutModel,
    _check_mc_dropout_model,
    _enable_only_dropout_layers,
)
from tests.test_utils import BreastCancerClassifier, get_classification_dataset


def test_enable_only_dropout_layers():
    model = BreastCancerClassifier(dropout_rate=0.5)
    model.eval()
    _enable_only_dropout_layers(model, p=0.25)
    assert model.dropout.training is True
    assert model.dropout.p == 0.25
    assert model.layer_1.training == model.layer_2.training == model.layer_3.training is False


def test_check_mc_dropout_model():
    _check_mc_dropout_model(BreastCancerClassifier, {"dropout_rate": 0.5})

    class Model(torch.nn.Module):
        def __init__(self):
            super(Model, self).__init__()

    with pytest.raises(ValueError) as err:
        _check_mc_dropout_model(Model, {})
    assert str(err.value) == """Model provided do not contain any torch.nn.Dropout modules, cannot apply MC Dropout"""


def test_MCDropoutEstimator():
    dataset = get_classification_dataset()
    # train the base model on mnist
    train_loader = dataset.get_train_loader()
    val_loader = dataset.get_validation_loader()
    model = LightningMCDropoutModel(
        BreastCancerClassifier,
        {"dropout_rate": 0.5},
        {"epochs": 1},
        eval_dropout_prob=0.2,
        n_estimators=3,
    )
    assert model.__class__.__name__ == "LightningMCDropoutModel"
    assert model.init_model().__class__.__name__ == "BreastCancerClassifier"

    model.train(train_loader)
    prediction = model(val_loader)
    assert prediction.size(0) == 3
    assert prediction.size(1) == len(dataset.validation_indices)
    assert isinstance(prediction, torch.Tensor)
