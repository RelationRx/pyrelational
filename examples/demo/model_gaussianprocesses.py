"""
This is a toy self-contained example of active learning on a regression
task with the active learning library

This example illustrates active learning with Gaussian Processes using
GPytorch.
"""

import logging

import gpytorch
import pytorch_lightning as pl
import torch

# Dataset and machine learning model
from examples.utils.datasets import DiabetesDataset  # noqa: E402

# Active Learning package
from pyrelational.data import DataManager
from pyrelational.models import LightningModel
from pyrelational.oracle import BenchmarkOracle
from pyrelational.pipeline import Pipeline
from pyrelational.strategies.regression import LeastConfidenceStrategy

# dataset
dataset = DiabetesDataset()
train_ds, val_ds, test_ds = torch.utils.data.random_split(dataset, [400, 22, 20])
train_indices = train_ds.indices
val_indices = val_ds.indices
test_indices = test_ds.indices


# Create GPytorch model
class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


# Creat PyL wrapper for GPytorch model
class PyLWrapper(pl.LightningModule):
    def __init__(self, train_x, train_y):
        super(PyLWrapper, self).__init__()
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        self.gpmodel = ExactGPModel(train_x, train_y, self.likelihood)
        self.criterion = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.gpmodel)

    def forward(self, x):
        return self.gpmodel(x)

    def generic_step(self, batch):
        x, y = batch
        x = self(x)
        loss = -self.criterion(x, y)
        return loss

    def training_step(self, batch, batch_idx):
        return self.generic_step(batch)

    def validation_step(self, batch, batch_idx):
        loss = self.generic_step(batch)
        self.log("loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        loss = self.generic_step(batch)
        self.log("test_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.gpmodel.parameters(), lr=0.1)


# Subclass LightningModel to handle GPytorch
class GPLightningModel(LightningModel):
    def __init__(self, model_class, model_config, trainer_config):
        super(GPLightningModel, self).__init__(model_class, model_config, trainer_config)

    def init_model(self, train_loader):
        for train_x, train_y in train_loader:
            return self.model_class(train_x=train_x, train_y=train_y, **self.model_config)

    def train(self, train_loader, valid_loader):
        trainer, ckpt_callback = self.init_trainer()
        model = self.init_model(train_loader)
        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=valid_loader)
        if valid_loader is not None:
            model.load_state_dict(torch.load(ckpt_callback.best_model_path)["state_dict"])
        self.current_model = model

    def __call__(self, loader):
        with torch.no_grad():
            self.current_model.gpmodel.eval()
            for x, _ in loader:
                return self.current_model(x)


model = GPLightningModel(model_class=PyLWrapper, model_config={}, trainer_config={"epochs": 1})

# data_manager and defining strategy
data_manager = DataManager(
    dataset=dataset,
    train_indices=train_indices,
    validation_indices=val_indices,
    test_indices=test_indices,
    loader_batch_size="full",
)  # all the labelled data points have to be used for training

# Set up strategy and rest of the pipeline
strategy = LeastConfidenceStrategy()
oracle = BenchmarkOracle()
pipeline = Pipeline(data_manager=data_manager, model=model, strategy=strategy, oracle=oracle)

# Remove lightning prints
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)

# performance with the full trainset labelled
pipeline.theoretical_performance()

# New data to be annotated, followed by an update of the data_manager and model
to_annotate = pipeline.active_learning_step(num_annotate=100)
pipeline.active_learning_update(indices=to_annotate, update_tag="Manual Update")

# Annotating data step by step until the trainset is fully annotated
pipeline.full_active_learning_run(num_annotate=100)
print(pipeline)
