import pytest
import torch

from pyrelational.models.ensemble_model import (
    GenericEnsembleModel,
    LightningEnsembleModel,
)
from tests.test_utils import BreastCancerClassifier, get_classification_dataset


def test_EnsembleEstimator():
    dataset = get_classification_dataset()
    train_loader = dataset.get_train_loader()
    val_loader = dataset.get_validation_loader()
    model = LightningEnsembleModel(BreastCancerClassifier, {}, {"epochs": 1}, n_estimators=4)
    assert model.__class__.__name__ == "LightningEnsembleModel"
    assert model.current_model is None
    assert isinstance(model.trainer_config, dict)
    assert isinstance(model.model_config, dict)

    with pytest.raises(ValueError) as err:
        model.test(val_loader)
        assert str(err.value) == "No current model, call 'train(train_loader, valid_loader)' to train the model first"

    model.train(train_loader)
    assert len(model.current_model) == 4

    prediction = model(val_loader)

    assert prediction.size(0) == 4
    assert prediction.size(1) == len(dataset.validation_indices)
    assert isinstance(prediction, torch.Tensor)
    assert isinstance(model.test(val_loader), dict)

    with pytest.raises(TypeError) as err:
        GenericEnsembleModel(BreastCancerClassifier, {}, {"epochs": 1}, n_estimators=4)
        assert (
            str(err.value) == "Can't instantiate abstract class GenericEnsembleModel with abstract methods test, train"
        )
