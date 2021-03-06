"""
This is a toy self-contained example of active learning on a classification
task with the active learning library

Here we give an example of defining your own custom AL strategy
"""

import logging

import torch

# Dataset and machine learning model
from examples.utils.datasets import BreastCancerDataset  # noqa: E402
from examples.utils.ml_models import BreastCancerClassification  # noqa: E402

# Active Learning package
from pyrelational.data import GenericDataManager
from pyrelational.models import LightningModel
from pyrelational.strategies.task_agnostic.relative_distance_strategy import (
    RelativeDistanceStrategy,
)

# dataset
dataset = BreastCancerDataset()
train_ds, val_ds, test_ds = torch.utils.data.random_split(dataset, [500, 30, 39])
train_indices = train_ds.indices
val_indices = val_ds.indices
test_indices = test_ds.indices

# model
model = LightningModel(model_class=BreastCancerClassification, model_config={}, trainer_config={"epochs": 4})

# data_manager and defining strategy
data_manager = GenericDataManager(
    dataset=dataset,
    train_indices=train_indices,
    validation_indices=val_indices,
    test_indices=test_indices,
    hit_ratio_at=5,
)

strategy = RelativeDistanceStrategy(data_manager=data_manager, model=model)

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
