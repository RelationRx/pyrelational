"""
Utility functions for scripting Active learning benchmark experiments where the model is a classifier.
"""

import os
from typing import Any, Dict, List, Optional, Type, TypeVar, Union

ModelType = TypeVar("ModelType")
E = TypeVar("E")

import numpy as np
import ray
import torch
from numpy.typing import NDArray

# Ray Tune
from ray import tune
from ray.train import RunConfig

# Scikit learn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import auc, balanced_accuracy_score, roc_auc_score
from torch.utils.data import DataLoader

# Pyrelational
from pyrelational.model_managers import ModelManager
from pyrelational.strategies.classification import (
    EntropyClassificationStrategy,
    LeastConfidenceStrategy,
    MarginalConfidenceStrategy,
    RatioConfidenceStrategy,
)
from pyrelational.strategies.task_agnostic import RandomAcquisitionStrategy


def get_strategy_from_string(strategy: str) -> Any:
    if strategy == "least_confidence":
        return LeastConfidenceStrategy()
    elif strategy == "entropy":
        return EntropyClassificationStrategy()
    elif strategy == "marginal_confidence":
        return MarginalConfidenceStrategy()
    elif strategy == "ratio_confidence":
        return RatioConfidenceStrategy()
    elif strategy == "random":
        return RandomAcquisitionStrategy()
    else:
        raise ValueError("Invalid strategy")


def numpy_collate(
    batch: List[Union[torch.Tensor, NDArray[Union[Any, np.float32, np.float64]]]]
) -> List[NDArray[Union[Any, np.float32, np.float64]]]:
    """Collate function for a Pytorch to Numpy DataLoader"""
    return [np.stack(el) for el in zip(*batch)]


# Wrapping the RFC with pyrelational's ModelManager
class SKRFC(ModelManager[RandomForestClassifier, RandomForestClassifier]):
    """
    Scikit learn RandomForestClassifier implementing the interface of our ModelManager
    for active learning.
    """

    def __init__(
        self, model_class: Type[RandomForestClassifier], model_config: Dict[str, Any], trainer_config: Dict[str, Any]
    ):
        super(SKRFC, self).__init__(model_class, model_config, trainer_config)

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
            metric = balanced_accuracy_score(y_hat, y)
            return {"test_metric": metric}

    def __call__(self, loader: DataLoader[Any]) -> Any:
        if not self.is_trained():
            raise ValueError("No current model, call 'train(X, y)' to train the model first")
        X, _ = next(iter(loader))
        model = self._current_model
        if model is None:
            raise ValueError("No current model, call 'train(X, y)' to train the model first")
        else:
            class_probabilities = model.predict_proba(X)
            return torch.FloatTensor(class_probabilities).unsqueeze(0)  # unsqueeze due to batch expectation


experiment_param_space = {
    "seed": tune.grid_search([1, 2, 3, 4, 5]),
    "strategy": tune.grid_search(["least_confidence", "entropy", "marginal_confidence", "ratio_confidence", "random"]),
}
