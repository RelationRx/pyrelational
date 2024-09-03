"""
Utility functions for scripting Active learning benchmark experiments where the model is a regressor.
"""

import os
from typing import Any, Dict, List, Optional, Type, TypeVar, Union

import numpy as np
import ray
import torch
from numpy.typing import NDArray

# Ray Tune
from ray import tune
from ray.train import RunConfig

# Scikit learn
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.metrics import mean_squared_error, r2_score
from torch.nn import Module
from torch.utils.data import DataLoader

# Pyrelational
from pyrelational.model_managers import ModelManager
from pyrelational.strategies.regression import (
    BALDStrategy,
    ExpectedImprovementStrategy,
    GreedyStrategy,
    ThompsonSamplingStrategy,
    UpperConfidenceBoundStrategy,
    VarianceReductionStrategy,
)
from pyrelational.strategies.regression.regression_strategy import RegressionStrategy
from pyrelational.strategies.task_agnostic import RandomAcquisitionStrategy

ModelType = TypeVar("ModelType", bound=Module)


def get_strategy_from_string(strategy: str) -> Any:
    if strategy == "bald":
        return BALDStrategy()
    elif strategy == "expected_improvement":
        return ExpectedImprovementStrategy()
    elif strategy == "greedy":
        return GreedyStrategy()
    elif strategy == "thompson_sampling":
        return ThompsonSamplingStrategy()
    elif strategy == "upper_confidence_bound":
        return UpperConfidenceBoundStrategy()
    elif strategy == "variance_reduction":
        return VarianceReductionStrategy()
    elif strategy == "random":
        return RandomAcquisitionStrategy()
    else:
        raise ValueError("Invalid strategy")


def numpy_collate(
    batch: List[Union[torch.Tensor, NDArray[Union[Any, np.float32, np.float64]]]]
) -> List[NDArray[Union[Any, np.float32, np.float64]]]:
    """Collate function for a Pytorch to Numpy DataLoader"""
    return [np.stack(el) for el in zip(*batch)]


# Wrapping the GPR with pyrelational's ModelManager
class GPR(ModelManager[GaussianProcessRegressor, GaussianProcessRegressor]):
    """
    Scikit learn GaussianProcessRegressor implementing the interface of our ModelManager
    for active learning.
    """

    def __init__(self, model_config: Dict[str, Any], trainer_config: Dict[str, Any]):
        super(GPR, self).__init__(GaussianProcessRegressor, model_config, trainer_config)

    def train(self, train_loader: DataLoader[Any], valid_loader: Optional[DataLoader[Any]] = None) -> None:
        train_x, train_y = next(iter(train_loader))
        estimator = self._init_model()
        estimator.fit(train_x, train_y)
        self._current_model = estimator

    def test(self, loader: DataLoader[Any]) -> Dict[str, float]:
        if not self.is_trained():
            raise ValueError("No current model, call 'train(X, y)' to train the model first")
        X, y = next(iter(loader))
        if self._current_model is None:
            raise ValueError("No current model, call 'train(X, y)' to train the model first")
        else:
            y_hat = self._current_model.predict(X)
            metric = mean_squared_error(y_hat, y)
            return {"test_metric": metric}

    def __call__(self, loader: DataLoader[Any]) -> Any:
        if not self.is_trained():
            raise ValueError("No current model, call 'train(X, y)' to train the model first")
        X, _ = next(iter(loader))
        model = self._current_model
        if model is None:
            raise ValueError("No current model, call 'train(X, y)' to train the model first")
        else:
            class_probabilities = model.sample_y(X, n_samples=10)
            return torch.FloatTensor(class_probabilities)


experiment_param_space = {
    "seed": tune.grid_search([1, 2, 3, 4, 5]),
    "strategy": tune.grid_search(
        [
            "bald",
            "expected_improvement",
            "greedy",
            "thompson_sampling",
            "upper_confidence_bound",
            "variance_reduction",
            "random",
        ]
    ),
}
