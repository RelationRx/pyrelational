"""
This is a toy self-contained on how to use Badge with the pyrelational package.

About BADGE algorithm: https://arxiv.org/abs/1906.03671
"""

import logging
from typing import List

import torch

# Dataset and machine learning model
from examples.utils.datasets import BreastCancerDataset  # noqa: E402
from examples.utils.ml_models import BreastCancerClassification  # noqa: E402

# Active Learning package
from pyrelational.data import DataManager
from pyrelational.informativeness import relative_distance
from pyrelational.models import LightningModel, ModelManager
from pyrelational.oracles import BenchmarkOracle
from pyrelational.pipeline import Pipeline
from pyrelational.strategies.abstract_strategy import Strategy

# dataset
dataset = BreastCancerDataset()
train_ds, val_ds, test_ds = torch.utils.data.random_split(dataset, [500, 30, 39])
train_indices = train_ds.indices
val_indices = val_ds.indices
test_indices = test_ds.indices


# model
class BadgeLightningModel(LightningModel):
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
        if self.current_model is None:
            raise ValueError(
                """
                    Trying to query gradients of an untrained model,
                    train model before calling get_gradients.
                """
            )

        model = self.current_model
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


model = BadgeLightningModel(model_class=BreastCancerClassification, model_config={}, trainer_config={"epochs": 5})

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

    def __call__(self, num_annotate: int, data_manager: DataManager, model: ModelManager) -> List[int]:
        """
        :param num_annotate: Number of samples to label
        :return: indices of samples to label
        """
        l_loader = data_manager.get_labelled_loader()
        u_loader = data_manager.get_unlabelled_loader()
        valid_loader = data_manager.get_validation_loader()
        model.train(l_loader, valid_loader)
        u_grads = model.get_gradients(u_loader)
        l_grads = model.get_gradients(l_loader)
        scores = relative_distance(u_grads, l_grads)
        ixs = torch.argsort(scores, descending=True).tolist()
        return [data_manager.u_indices[i] for i in ixs[:num_annotate]]


# Set the instantiated custom model and strategy into the Pipeline object
strategy = BadgeStrategy()
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
