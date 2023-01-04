"""
This is a toy self-contained example of active learning on a classification
task with the active learning library

It illustrates the ensemble method.
"""

import logging

import torch

# Pytorch
from torchvision import datasets, transforms

# Dataset and machine learning model
from examples.utils.ml_models import MnistClassification

# Active Learning package
from pyrelational.data import DataManager
from pyrelational.models import LightningEnsembleModel
from pyrelational.strategies.classification import LeastConfidenceStrategy

# dataset
dataset = datasets.FashionMNIST(root="data", train=True, download=True, transform=transforms.ToTensor())

dataset = [dataset[i] for i in range(10000)]

train_ds, val_ds, test_ds = torch.utils.data.random_split(dataset, [9000, 500, 500])
train_indices = train_ds.indices
val_indices = val_ds.indices
test_indices = test_ds.indices

# model
model = LightningEnsembleModel(
    model_class=MnistClassification, model_config={}, trainer_config={"epochs": 4}, n_estimators=5
)

# data_manager and defining strategy
data_manager = DataManager(
    dataset=dataset,
    train_indices=train_indices,
    validation_indices=val_indices,
    test_indices=test_indices,
    loader_batch_size=1000,
)

strategy = LeastConfidenceStrategy(data_manager=data_manager, model=model)

# Remove lightning prints
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)

# performance with the full trainset labelled
strategy.theoretical_performance()

# New data to be annotated, followed by an update of the data_manager and model
to_annotate = strategy.active_learning_step(num_annotate=1000)
strategy.active_learning_update(indices=to_annotate, update_tag="Manual Update")

# Annotating data step by step until the trainset is fully annotated
strategy.full_active_learning_run(num_annotate=1000)
print(strategy)
