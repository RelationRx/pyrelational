"""
This is a toy self-contained example of active learning on a classification
task with the active learning library

This is an example of using pyrelational with a RandomForestClassifier from scikit learn
"""

import numpy as np
import torch

# Scikit learn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from torch.utils.data import Dataset

from examples.utils.datasets import BreastCancerDataset  # noqa: E402
from pyrelational.data import GenericDataManager

# pyrelational
from pyrelational.models import GenericModel
from pyrelational.strategies.classification import LeastConfidenceStrategy


def numpy_collate(batch):
    """Collate function for a Pytorch to Numpy DataLoader"""
    return [np.stack(el) for el in zip(*batch)]


def get_breastcancer_data_manager():
    ds = BreastCancerDataset()
    train_ds, valid_ds, test_ds = torch.utils.data.random_split(ds, [400, 100, 69])
    train_indices = train_ds.indices
    valid_indices = valid_ds.indices
    test_indices = test_ds.indices

    return GenericDataManager(
        ds,
        train_indices=train_indices,
        validation_indices=valid_indices,
        test_indices=test_indices,
        loader_batch_size="full",
        loader_collate_fn=numpy_collate,
    )


# Wrapping the RFC with pyrelational's GenericModel
class SKRFC(GenericModel):
    """
    Scikit learn RandomForestClassifier implementing the interface of our GenericModel
    for active learning.
    """

    def __init__(self, model_class, model_config, trainer_config):
        super(SKRFC, self).__init__(model_class, model_config, trainer_config)

    def train(self, train_loader, valid_loader):
        train_x, train_y = next(iter(train_loader))
        estimator = self.init_model()
        estimator.fit(train_x, train_y, **trainer_config)
        self.current_model = estimator

    def test(self, loader):
        if self.current_model is None:
            raise ValueError("No current model, call 'train(X, y)' to train the model first")
        X, y = next(iter(loader))
        y_hat = self.current_model.predict(X)
        acc = accuracy_score(y_hat, y)
        return {"test_acc": acc}

    def __call__(self, loader):
        if self.current_model is None:
            raise ValueError("No current model, call 'train(X, y)' to train the model first")
        X, _ = next(iter(loader))
        model = self.current_model
        class_probabilities = model.predict_proba(X)
        return torch.FloatTensor(class_probabilities).unsqueeze(0)  # unsqueeze due to batch expectation


data_manager = get_breastcancer_data_manager()
model_config = {"n_estimators": 10, "bootstrap": False}
trainer_config = {}
model = SKRFC(RandomForestClassifier, model_config, trainer_config)

# Run active learning strategy
al_strategy = LeastConfidenceStrategy(data_manager, model)

# performance with the full trainset labelled
al_strategy.theoretical_performance()

# New data to be annotated, followed by an update of the data_manager and model
to_annotate = al_strategy.active_learning_step(num_annotate=100)
al_strategy.active_learning_update(to_annotate, oracle_interface=None, update_tag="Manual Update")

# Annotating data step by step until the trainset is fully annotated
al_strategy.full_active_learning_run(num_annotate=100)
print(al_strategy)
