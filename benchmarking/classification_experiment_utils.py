"""
Utility functions for scripting Active learning benchmark experiments where the model is a classifier.
"""

import torch

# Scikit learn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, auc, roc_auc_score

# Pyrelational
from pyrelational.model_managers import ModelManager
from pyrelational.strategies.classification import (LeastConfidenceStrategy, 
                                                    EntropyClassificationStrategy, 
                                                    MarginalConfidenceStrategy,
                                                    RatioConfidenceStrategy)
from pyrelational.strategies.task_agnostic import RandomAcquisitionStrategy

# Ray Tune
from ray import tune
from ray.train import RunConfig
import ray
import os

def get_strategy_from_string(strategy):
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

# Wrapping the RFC with pyrelational's ModelManager
class SKRFC(ModelManager):
    """
    Scikit learn RandomForestClassifier implementing the interface of our ModelManager
    for active learning.
    """

    def __init__(self, model_class, model_config, trainer_config):
        super(SKRFC, self).__init__(model_class, model_config, trainer_config)

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
        metric = balanced_accuracy_score(y_hat, y)
        return {"test_metric": metric}

    def __call__(self, loader):
        if not self.is_trained():
            raise ValueError("No current model, call 'train(X, y)' to train the model first")
        X, _ = next(iter(loader))
        model = self._current_model
        class_probabilities = model.predict_proba(X)
        return torch.FloatTensor(class_probabilities).unsqueeze(0)  # unsqueeze due to batch expectation

experiment_param_space = {
    "seed": tune.grid_search([1,2,3,4,5]),
    "strategy": tune.grid_search(["least_confidence", "entropy", "marginal_confidence", "ratio_confidence"])
}
