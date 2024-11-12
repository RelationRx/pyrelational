"""
Utility functions for scripting Active learning benchmark experiments where the model is a classifier.
"""

import random
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np
import torch
from numpy.typing import NDArray

# Ray Tune
from ray import tune

# Scikit learn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
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
            metric = balanced_accuracy_score(y, y_hat)
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


class LogisticRegressor(ModelManager[Any, Any]):
    """
    Scikit learn LogisticRegression implementing the interface of our ModelManager
    for active learning.
    """

    def __init__(self, model_class: Type[Any], model_config: Dict[str, Any], trainer_config: Dict[str, Any]):
        super(LogisticRegressor, self).__init__(model_class, model_config, trainer_config)

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
            metric = balanced_accuracy_score(y, y_hat)
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


def pick_one_sample_per_class(dataset: Any, train_indices: NDArray[Union[Any, np.int64]]) -> List[int]:
    """Randomly pick one sample per class in the training subset of dataset
    and return their index in the dataset. This is used for defining the
    initial state of the labelled subset in cold start active learning tasks
    """
    class2idx = defaultdict(list)
    for idx in train_indices:
        idx_class = int(dataset[idx][1])
        class2idx[idx_class].append(idx)

    class_reps = []
    for idx_class in class2idx.keys():
        random_class_idx = random.choice(class2idx[idx_class])
        class_reps.append(random_class_idx)

    return class_reps


def make_class_stratified_train_val_test_split(dataset: Any, k: int) -> Tuple[
    NDArray[Union[Any, np.float32, np.float64]],
    NDArray[Union[Any, np.float32, np.float64]],
    NDArray[Union[Any, np.float32, np.float64]],
]:
    """Return train, val, test indices that respect a class-stratified split"""
    skf = StratifiedKFold(n_splits=k, shuffle=True)
    X = np.array(range(len(dataset)))
    y = np.array([dataset[idx][1] for idx in X])
    for train_index, test_index in skf.split(X, y):
        train_indices, test_indices = X[train_index], X[test_index]
        break
    val_indices = train_indices[: len(train_indices) // 5]
    train_indices = train_indices[len(train_indices) // 5 :]
    return train_indices, val_indices, test_indices


experiment_param_space = {
    "seed": tune.grid_search([1, 2, 3, 4, 5]),
    "strategy": tune.grid_search(["least_confidence", "entropy", "marginal_confidence", "ratio_confidence", "random"]),
}
