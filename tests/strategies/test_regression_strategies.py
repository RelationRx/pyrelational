from pyrelational.models.mcdropout_model import LightningMCDropoutModel
from pyrelational.strategies.regression import (
    BALDStrategy,
    ExpectedImprovementStrategy,
    GreedyStrategy,
    LeastConfidenceStrategy,
    SoftBALDStrategy,
    ThompsonSamplingStrategy,
    UpperConfidenceBoundStrategy,
)
from tests.test_utils import DiabetesRegression, get_regression_dataset


def test_regression_strategies():
    model = get_model()
    data_manager = get_regression_dataset()

    LeastConfidenceStrategy(model=model, data_manager=data_manager).active_learning_step(num_annotate=100)
    UpperConfidenceBoundStrategy(
        model=model,
        data_manager=data_manager,
        kappa=2,
    ).active_learning_step(num_annotate=100)
    ThompsonSamplingStrategy(model=model, data_manager=data_manager).active_learning_step(num_annotate=100)
    GreedyStrategy(model=model, data_manager=data_manager).active_learning_step(num_annotate=100)
    ExpectedImprovementStrategy(model=model, data_manager=data_manager).active_learning_step(num_annotate=100)
    BALDStrategy(model=model, data_manager=data_manager).active_learning_step(num_annotate=100)
    SoftBALDStrategy(model=model, data_manager=data_manager).active_learning_step(num_annotate=100)


def get_model():
    model = LightningMCDropoutModel(DiabetesRegression, {}, {"epochs": 5, "gpus": 0})
    return model
