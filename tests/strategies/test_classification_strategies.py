"""Unit tests for active learning manager
"""
import pytest

from pyrelational.models.mcdropout_model import LightningMCDropoutModel
from pyrelational.oracle import BenchmarkOracle
from pyrelational.pipeline import Pipeline
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
    al_strategy = LeastConfidenceStrategy()
    pipeline = Pipeline(data_manager=gdm, model=model, strategy=al_strategy)
    pipeline.theoretical_performance()
    assert "full" in pipeline.performances

    result = pipeline.current_performance()
    assert len(list(pipeline.performances.keys())) == 1
    assert result["test_loss"] > pipeline.performances["full"]["test_loss"]


def test_full_active_learning_run():
    gdm = get_classification_dataset(hit_ratio_at=5)
    model = LightningMCDropoutModel(BreastCancerClassifier, {"ensemble_size": 3}, {"epochs": 1})
    al_strategy = LeastConfidenceStrategy()
    oracle = BenchmarkOracle()
    pipeline = Pipeline(data_manager=gdm, model=model, strategy=al_strategy, oracle=oracle)
    pipeline.theoretical_performance()

    pipeline.full_active_learning_run(num_annotate=200)
    # Test performance history data frame
    df = pipeline.performance_history()
    print(df)
    assert df.shape == (3, 3)
    assert len(pipeline.data_manager.l_indices) == len(gdm.train_indices)
    assert len(pipeline.data_manager.u_indices) == 0
    assert {"full", 0, 1, 2} == set(list(pipeline.performances.keys()))
    for k in {"full", 0, 1, 2}:
        assert "hit_ratio" in pipeline.performances[k].keys()


# # TODO: Move these tests to the oracle
# def test_update_annotations():
#     gdm = get_classification_dataset()
#     model = LightningMCDropoutModel(BreastCancerClassifier, {"ensemble_size": 3}, {"epochs": 1})
#     al_strategy = LeastConfidenceStrategy()
#     pipeline = Pipeline(data_manager=gdm, model=model, strategy=al_strategy)
#     pipeline.theoretical_performance()

#     random_u_sindex = gdm.u_indices[0]
#     len_gdm_l = len(gdm.l_indices)
#     len_gdm_u = len(gdm.u_indices)

#     pipeline.update_annotations([random_u_sindex])
#     assert random_u_sindex in gdm.l_indices
#     assert len(gdm.l_indices) > len_gdm_l
#     assert len(gdm.u_indices) < len_gdm_u


def test_get_percentage_labelled():
    gdm = get_classification_dataset()
    model = LightningMCDropoutModel(BreastCancerClassifier, {"ensemble_size": 3}, {"epochs": 1})
    al_strategy = LeastConfidenceStrategy()
    pipeline = Pipeline(data_manager=gdm, model=model, strategy=al_strategy)
    percentage = pipeline.percentage_labelled
    assert percentage == pytest.approx(10, 5)


def test_get_dataset_size():
    gdm = get_classification_dataset()
    model = LightningMCDropoutModel(BreastCancerClassifier, {"ensemble_size": 3}, {"epochs": 1})
    al_strategy = LeastConfidenceStrategy()
    pipeline = Pipeline(data_manager=gdm, model=model, strategy=al_strategy)
    al_ds_size = pipeline.dataset_size
    assert al_ds_size <= 60000


def test_strategies():
    gdm = get_classification_dataset()
    model = LightningMCDropoutModel(BreastCancerClassifier, {"ensemble_size": 3}, {"epochs": 1})

    mcs = MarginalConfidenceStrategy()
    rcs = RatioConfidenceStrategy()
    ecs = EntropyClassificationStrategy()

    pipeline_mcs = Pipeline(data_manager=gdm, model=model, strategy=mcs)
    pipeline_rcs = Pipeline(data_manager=gdm, model=model, strategy=rcs)
    pipeline_ecs = Pipeline(data_manager=gdm, model=model, strategy=ecs)

    pipeline_mcs.current_performance()
    pipeline_rcs.current_performance()
    pipeline_ecs.current_performance()
