from pyrelational.models.mcdropout_model import LightningMCDropoutModel
from pyrelational.pipeline import GenericPipeline
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
    lcs = LeastConfidenceStrategy()
    ucbs = UpperConfidenceBoundStrategy(kappa=2)
    tss = ThompsonSamplingStrategy()
    gs = GreedyStrategy()
    eis = ExpectedImprovementStrategy()
    bs = BALDStrategy()
    sbs = SoftBALDStrategy()

    GenericPipeline(data_manager=data_manager, model=model, strategy=lcs).active_learning_step(num_annotate=5)
    GenericPipeline(data_manager=data_manager, model=model, strategy=ucbs).active_learning_step(num_annotate=5)
    GenericPipeline(data_manager=data_manager, model=model, strategy=tss).active_learning_step(num_annotate=5)
    GenericPipeline(data_manager=data_manager, model=model, strategy=gs).active_learning_step(num_annotate=5)
    GenericPipeline(data_manager=data_manager, model=model, strategy=eis).active_learning_step(num_annotate=5)
    GenericPipeline(data_manager=data_manager, model=model, strategy=bs).active_learning_step(num_annotate=5)
    GenericPipeline(data_manager=data_manager, model=model, strategy=sbs).active_learning_step(num_annotate=5)


def get_model():
    model = LightningMCDropoutModel(DiabetesRegression, {}, {"epochs": 5, "gpus": 0})
    return model
