"""
This is a toy self-contained on how to use Badge with the pyrelational package.

About BADGE algorithm: https://arxiv.org/abs/1906.03671
"""

import logging
from typing import List

import torch

# Dataset and machine learning model
from utils.datasets import BreastCancerDataset
from utils.ml_models import BreastCancerClassification

# Active Learning package
from pyrelational.data_managers import DataManager
from pyrelational.informativeness import relative_distance
from pyrelational.model_managers import LightningModelManager, ModelManager
from pyrelational.oracles import BenchmarkOracle
from pyrelational.pipeline import Pipeline
from pyrelational.strategies.abstract_strategy import Strategy

# dataset
dataset = BreastCancerDataset()
train_ds, val_ds, test_ds = torch.utils.data.random_split(dataset, [500, 30, 39])
train_indices = train_ds.indices
val_indices = val_ds.indices
test_indices = test_ds.indices


# model_manager
class BadgeLightningModel(LightningModelManager):
    """Model compatible with BADGE strategy"""

    def __init__(self, model_class, model_config, trainer_config):
        super(BadgeLightningModel, self).__init__(model_class, model_config, trainer_config)

    def get_gradients(self, loader):
        """
        Get gradients for each sample in dataloader as outlined in BADGE paper.

        Assumes the last layer is a linear layer and return_penultimate_embed/criterion is defined in the model class
        :param loader: dataloader
        :return: tensor of gradients for each sample
        """
        if not self.is_trained():
            raise ValueError(
                """
                    Trying to query gradients of an untrained model,
                    train model before calling get_gradients.
                """
            )

        model = self._current_model
        model.eval()
        gradients = []
        for x, _ in loader:
            model.zero_grad()
            logits = model(x)
            class_preds = torch.argmax(logits, dim=1)
            loss = model.criterion(logits, class_preds)  # assumes criterion is defined in model class
            e = model.return_penultimate_embed(x)
            # find gradients of bias in last layer
            bias_grad = torch.autograd.grad(loss, logits)[0]
            # find gradients of weights in last layer
            weights_grad = torch.einsum("be,bc -> bec", e, bias_grad)
            gradients.append(torch.cat([weights_grad.detach().cpu(), bias_grad.unsqueeze(1).detach().cpu()], 1))

        return torch.cat(gradients, 0)


model_manager = BadgeLightningModel(
    model_class=BreastCancerClassification, model_config={}, trainer_config={"epochs": 5}
)

# data_manager and defining strategy
data_manager = DataManager(
    dataset=dataset,
    train_indices=train_indices,
    validation_indices=val_indices,
    test_indices=test_indices,
    loader_batch_size=16,
)


class BadgeStrategy(Strategy):
    """Implementation of BADGE strategy."""

    def __init__(self):
        super(BadgeStrategy, self).__init__()

    def __call__(self, num_annotate: int, data_manager: DataManager, model_manager: ModelManager) -> List[int]:
        """
        :param num_annotate: Number of samples to label
        :return: indices of samples to label
        """
        l_loader = data_manager.get_labelled_loader()
        u_loader = data_manager.get_unlabelled_loader()
        valid_loader = data_manager.get_validation_loader()
        model_manager.train(l_loader, valid_loader)
        u_grads = model_manager.get_gradients(u_loader)
        l_grads = model_manager.get_gradients(l_loader)
        scores = relative_distance(u_grads, l_grads)
        ixs = torch.argsort(scores, descending=True).tolist()
        return [data_manager.u_indices[i] for i in ixs[:num_annotate]]


# Set the instantiated custom model and strategy into the Pipeline object
strategy = BadgeStrategy()
oracle = BenchmarkOracle()
pipeline = Pipeline(data_manager=data_manager, model_manager=model_manager, strategy=strategy, oracle=oracle)

# Remove lightning prints
logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)

# performance with the full trainset labelled
pipeline.compute_theoretical_performance()

# New data to be annotated, followed by an update of the data_manager and model
to_annotate = pipeline.step(num_annotate=100)
pipeline.query(indices=to_annotate)

# Annotating data step by step until the trainset is fully annotated
pipeline.run(num_annotate=100)
print(pipeline)
