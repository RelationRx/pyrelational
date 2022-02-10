"""Unit tests for active learning manager
"""
import pytest

from pyrelational.models.mcdropout_model import LightningMCDropoutModel
from pyrelational.strategies.classification import (
    EntropyClassificationStrategy,
    LeastConfidenceStrategy,
    MarginalConfidenceStrategy,
    RatioConfidenceStrategy,
)
from tests.test_utils import BreastCancerClassifier, get_classification_dataset


def test_performances():
    """
    Testing theoretical and current performance returns
    """
    gdm = get_classification_dataset()
    model = LightningMCDropoutModel(BreastCancerClassifier, {"ensemble_size": 3}, {"epochs": 1})
    al_manager = LeastConfidenceStrategy(data_manager=gdm, model=model)
    al_manager.theoretical_performance()
    assert "full" in al_manager.performances

    result = al_manager.current_performance()
    assert len(list(al_manager.performances.keys())) == 1
    assert result["test_loss"] > al_manager.performances["full"]["test_loss"]


def test_full_active_learning_run():
    gdm = get_classification_dataset()
    model = LightningMCDropoutModel(BreastCancerClassifier, {"ensemble_size": 3}, {"epochs": 1})
    al_manager = LeastConfidenceStrategy(data_manager=gdm, model=model)
    al_manager.theoretical_performance()

    al_manager.full_active_learning_run(num_annotate=200)
    # Test performance history data frame
    df = al_manager.performance_history()
    print(df)
    assert df.shape == (3, 2)
    assert len(al_manager.data_manager.l_indices) == len(gdm.train_indices)
    assert len(al_manager.data_manager.u_indices) == 0
    assert {"full", 0, 1, 2} == set(list(al_manager.performances.keys()))


def test_update_annotations():
    gdm = get_classification_dataset()
    model = LightningMCDropoutModel(BreastCancerClassifier, {"ensemble_size": 3}, {"epochs": 1})
    al_manager = LeastConfidenceStrategy(data_manager=gdm, model=model)
    al_manager.theoretical_performance()

    random_u_sindex = gdm.u_indices[0]
    len_gdm_l = len(gdm.l_indices)
    len_gdm_u = len(gdm.u_indices)

    al_manager.update_annotations([random_u_sindex])
    assert random_u_sindex in gdm.l_indices
    assert len(gdm.l_indices) > len_gdm_l
    assert len(gdm.u_indices) < len_gdm_u


def test_get_percentage_labelled():
    gdm = get_classification_dataset()
    model = LightningMCDropoutModel(BreastCancerClassifier, {"ensemble_size": 3}, {"epochs": 1})
    al_manager = LeastConfidenceStrategy(data_manager=gdm, model=model)
    percentage = al_manager.percentage_labelled
    assert percentage == pytest.approx(0.1, 0.05)


def test_get_dataset_size():
    gdm = get_classification_dataset()
    model = LightningMCDropoutModel(BreastCancerClassifier, {"ensemble_size": 3}, {"epochs": 1})
    al_manager = LeastConfidenceStrategy(
        data_manager=gdm,
        model=model,
    )
    al_ds_size = al_manager.dataset_size
    assert al_ds_size <= 60000


def test_strategies():
    gdm = get_classification_dataset()
    model = LightningMCDropoutModel(BreastCancerClassifier, {"ensemble_size": 3}, {"epochs": 1})
    MarginalConfidenceStrategy(data_manager=gdm, model=model).current_performance()
    RatioConfidenceStrategy(data_manager=gdm, model=model).current_performance()
    EntropyClassificationStrategy(data_manager=gdm, model=model).current_performance()
