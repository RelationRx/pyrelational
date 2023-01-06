from sklearn.cluster import AgglomerativeClustering

from pyrelational.models.lightning_model import LightningModel
from pyrelational.pipeline import Pipeline
from pyrelational.strategies.task_agnostic import (
    RandomAcquisitionStrategy,
    RelativeDistanceStrategy,
    RepresentativeSamplingStrategy,
)
from tests.test_utils import DiabetesRegressionModel, get_regression_dataset


def test_diversity_strategies():
    model = get_model()
    data_manager = get_regression_dataset()

    rds = RelativeDistanceStrategy()
    rds_cosine = RelativeDistanceStrategy()
    ras = RandomAcquisitionStrategy()
    rss = RepresentativeSamplingStrategy(clustering_method="AffinityPropagation")

    agg = AgglomerativeClustering(n_clusters=10)
    rss_agg = RepresentativeSamplingStrategy(clustering_method=agg)

    Pipeline(data_manager=data_manager, model=model, strategy=rds).active_learning_step(num_annotate=5)
    Pipeline(data_manager=data_manager, model=model, strategy=rds_cosine).active_learning_step(
        num_annotate=5, metric="cosine"
    )
    Pipeline(data_manager=data_manager, model=model, strategy=ras).active_learning_step(num_annotate=5)
    Pipeline(data_manager=data_manager, model=model, strategy=rss).active_learning_step(num_annotate=5)
    Pipeline(data_manager=data_manager, model=model, strategy=rss_agg).active_learning_step()


def get_model():
    model = LightningModel(DiabetesRegressionModel, {}, {"epochs": 5, "gpus": 0})
    return model
