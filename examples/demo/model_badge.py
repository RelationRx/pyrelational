"""
This is a toy self-contained on how to use Badge with the pyrelational package.

About BADGE algorithm: https://arxiv.org/abs/1906.03671
"""

import logging

import torch

# Dataset and machine learning model
from examples.utils.datasets import DiabetesDataset  # noqa: E402
from examples.utils.ml_models import DiabetesRegression  # noqa: E402

# Active Learning package
from pyrelational.data import GenericDataManager
from pyrelational.informativeness import relative_distance
from pyrelational.models import LightningModel
from pyrelational.strategies.generic_al_strategy import GenericActiveLearningStrategy

# dataset
dataset = DiabetesDataset()
train_ds, val_ds, test_ds = torch.utils.data.random_split(dataset, [400, 22, 20])
train_indices = train_ds.indices
val_indices = val_ds.indices
test_indices = test_ds.indices


# model


class BadgeLightningModel(LightningModel):
    def __init__(self, model_class, model_config, trainer_config):
        super(BadgeLightningModel, self).__init__(model_class, model_config, trainer_config)

    def get_gradients(self, loader):
        gradients = []
        model = self.init_model()
        for x, y in loader:  # loader should have batch size of 1
            model.zero_grad()
            pred = model(x)
            loss = model.criterion(y, pred)  # assumes criterion is defined in model class
            loss.backward()
            gradients.append(
                torch.cat([w.grad.flatten() for w in list(model.parameters())[-3:-1]], 0)
            )  # taking gradients from last layers (bias and weights)
        return torch.stack(gradients)


model = BadgeLightningModel(model_class=DiabetesRegression, model_config={}, trainer_config={"epochs": 5})

# data_manager and defining strategy
data_manager = GenericDataManager(
    dataset=dataset,
    train_indices=train_indices,
    validation_indices=val_indices,
    test_indices=test_indices,
    loader_batch_size=1,
)  # make sure batch size is 1 for gradient estimates


class BadgeStrategy(GenericActiveLearningStrategy):
    def __init__(self, data_manager, model):
        super(BadgeStrategy, self).__init__(data_manager, model)

    def active_learning_step(self, num_annotate):
        u_grads = self.model.get_gradients(self.u_loader)
        l_grads = self.model.get_gradients(self.l_loader)
        scores = relative_distance(u_grads, l_grads)
        ixs = torch.argsort(scores, descending=True).tolist()
        return [self.u_indices[i] for i in ixs[:num_annotate]]


strategy = BadgeStrategy(data_manager=data_manager, model=model)

# Remove lightning prints
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)

# performance with the full trainset labelled
strategy.theoretical_performance()

# New data to be annotated, followed by an update of the data_manager and model
to_annotate = strategy.active_learning_step(num_annotate=100)
strategy.active_learning_update(indices=to_annotate, update_tag="Manual Update")

# Annotating data step by step until the trainset is fully annotated
strategy.full_active_learning_run(num_annotate=100)
print(strategy)
