import os
import sys

from sklearn.cluster import AgglomerativeClustering

from pyrelational.models.lightning_model import LightningModel
from pyrelational.strategies.task_agnostic import (
    RandomAcquisitionStrategy,
    RelativeDistanceStrategy,
    RepresentativeSamplingStrategy,
)
from tests.test_utils import DiabetesRegression, get_regression_dataset


def test_diversity_strategies():
    model = get_model()
    data_manager = get_regression_dataset()

    RelativeDistanceStrategy(model=model, data_manager=data_manager).active_learning_step(num_annotate=100)
    RelativeDistanceStrategy(model=model, data_manager=data_manager).active_learning_step(
        num_annotate=100, metric="cosine"
    )
    RandomAcquisitionStrategy(model=model, data_manager=data_manager).active_learning_step(num_annotate=100)
    RepresentativeSamplingStrategy(
        model=model, data_manager=data_manager, clustering_method="AffinityPropagation"
    ).active_learning_step(num_annotate=100)
    agg = AgglomerativeClustering(n_clusters=10)
    RepresentativeSamplingStrategy(model=model, data_manager=data_manager, clustering_method=agg).active_learning_step()


def get_model():
    model = LightningModel(DiabetesRegression, {}, {"epochs": 5, "gpus": 0})
    return model
