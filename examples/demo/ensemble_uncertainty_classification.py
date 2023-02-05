"""
This is a toy self-contained example of active learning on a classification
task with the active learning library

It illustrates the ensemble method.
"""

# User imports
import logging

import torch
from torchvision import datasets, transforms

# Dataset and machine learning model
from examples.utils.ml_models import MnistClassification

# Active Learning package
from pyrelational.data_managers import DataManager
from pyrelational.model_managers import LightningEnsembleModelManager
from pyrelational.oracles import BenchmarkOracle
from pyrelational.pipeline import Pipeline
from pyrelational.strategies.classification import LeastConfidenceStrategy

# dataset
dataset = datasets.FashionMNIST(root="data", train=True, download=True, transform=transforms.ToTensor())
dataset = torch.utils.data.Subset(dataset, indices=range(10000))

train_ds, val_ds, test_ds = torch.utils.data.random_split(dataset, [9000, 500, 500])
train_indices = train_ds.indices
val_indices = val_ds.indices
test_indices = test_ds.indices

# model
model = LightningEnsembleModelManager(
    model_class=MnistClassification, model_config={}, trainer_config={"epochs": 5}, n_estimators=5
)

# data_manager and defining strategy
data_manager = DataManager(
    dataset=dataset,
    train_indices=train_indices,
    validation_indices=val_indices,
    test_indices=test_indices,
    loader_batch_size=1000,
)

# Set up active learning pipeline
strategy = LeastConfidenceStrategy()
oracle = BenchmarkOracle()
pipeline = Pipeline(data_manager=data_manager, model=model, strategy=strategy, oracle=oracle)

# Remove lightning prints
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)

# See performance with the full trainset labelled
pipeline.theoretical_performance()

# New data to be annotated, followed by an update of the data_manager and model
to_annotate = pipeline.active_learning_step(num_annotate=1000)
pipeline.active_learning_update(indices=to_annotate, update_tag="Manual Update")

# Annotating data step by step until the trainset is fully annotated
pipeline.full_active_learning_run(num_annotate=1000)

# Pretty printed summary of the components in the pipeline along with annotation/performance history
print(pipeline)
