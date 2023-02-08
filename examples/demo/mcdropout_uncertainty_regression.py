"""
This is a toy self-contained example of active learning on a regression
task with the active learning library

This example will use uncertainty arising from the standard deviation of the
predictive distribution obtained via MCDropout
"""

import logging

import torch

# Dataset and machine learning model
from examples.utils.datasets import DiabetesDataset  # noqa: E402
from examples.utils.ml_models import DiabetesRegression  # noqa: E402

# Active Learning package
from pyrelational.data_managers import DataManager
from pyrelational.model_managers import LightningMCDropoutModelManager
from pyrelational.oracles import BenchmarkOracle
from pyrelational.pipeline import Pipeline
from pyrelational.strategies.regression import LeastConfidenceStrategy

# dataset
dataset = DiabetesDataset()
train_ds, val_ds, test_ds = torch.utils.data.random_split(dataset, [400, 22, 20])
train_indices = train_ds.indices
val_indices = val_ds.indices
test_indices = test_ds.indices

# model_manager
model_manager = LightningMCDropoutModelManager(
    model_class=DiabetesRegression, model_config={}, trainer_config={"epochs": 4}
)

# data_manager and defining strategy
data_manager = DataManager(
    dataset=dataset, train_indices=train_indices, validation_indices=val_indices, test_indices=test_indices
)


strategy = LeastConfidenceStrategy()
oracle = BenchmarkOracle()
pipeline = Pipeline(data_manager=data_manager, model_manager=model_manager, strategy=strategy, oracle=oracle)

# Remove lightning prints
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)

# performance with the full trainset labelled
pipeline.compute_theoretical_performance()

# New data to be annotated, followed by an update of the data_manager and model
to_annotate = pipeline.step(num_annotate=100)
pipeline.query(indices=to_annotate)

# Annotating data step by step until the trainset is fully annotated
pipeline.full_active_learning_run(num_annotate=100)
print(pipeline)
