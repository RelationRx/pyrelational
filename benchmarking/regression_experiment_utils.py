"""
Utility functions for scripting Active learning benchmark experiments where the model is a regressor.
"""

import torch
import numpy as np

# Scikit learn
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Pyrelational
from pyrelational.model_managers import ModelManager
from pyrelational.strategies.regression import (BALDStrategy,
                                ExpectedImprovementStrategy,
                                GreedyStrategy,
                                ThompsonSamplingStrategy,
                                UpperConfidenceBoundStrategy,
                                VarianceReductionStrategy)
from pyrelational.strategies.task_agnostic import RandomAcquisitionStrategy

# Ray Tune
from ray import tune
from ray.train import RunConfig
import ray
import os

def get_strategy_from_string(strategy):
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
    
def numpy_collate(batch):
    """Collate function for a Pytorch to Numpy DataLoader"""
    return [np.stack(el) for el in zip(*batch)]

# Wrapping the GPR with pyrelational's ModelManager
class GPR(ModelManager):
    """
    Scikit learn GaussianProcessRegressor implementing the interface of our ModelManager
    for active learning.
    """

    def __init__(self, model_config, trainer_config):
        super(GPR, self).__init__(GaussianProcessRegressor, model_config, trainer_config)

    def train(self, train_loader, valid_loader):
        train_x, train_y = next(iter(train_loader))
        estimator = self._init_model()
        estimator.fit(train_x, train_y)
        self._current_model = estimator

    def test(self, loader):
        if not self.is_trained():
            raise ValueError("No current model, call 'train(X, y)' to train the model first")
        X, y = next(iter(loader))
        y_hat = self._current_model.predict(X)
        metric = mean_squared_error(y_hat, y)
        return {"test_metric": metric}

    def __call__(self, loader):
        if not self.is_trained():
            raise ValueError("No current model, call 'train(X, y)' to train the model first")
        X, _ = next(iter(loader))
        model = self._current_model
        class_probabilities = model.sample_y(X, n_samples=10)
        return torch.FloatTensor(class_probabilities) 
    
experiment_param_space = {
    "seed": tune.grid_search([1,2,3,4,5]),
    "strategy": tune.grid_search(["bald", "expected_improvement", "greedy", "thompson_sampling", "upper_confidence_bound", "variance_reduction", "random"])
}