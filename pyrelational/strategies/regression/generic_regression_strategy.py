from abc import ABC

import torch

from pyrelational.data import DataManager
from pyrelational.models import ModelManager
from pyrelational.strategies.generic_al_strategy import GenericActiveLearningStrategy


class GenericRegressionStrategy(GenericActiveLearningStrategy, ABC):
    """A base active learning strategy class for regression in which the top n indices,
    according to user-specified scoring function, are queried at each iteration"""

    def __init__(self):
        super(GenericRegressionStrategy, self).__init__()
        self.scoring_fn = NotImplementedError

    def active_learning_step(self, num_annotate: int, data_manager: DataManager, model: ModelManager):
        output = self.train_and_infer(data_manager=data_manager, model=model)
        scores = self.scoring_fn(x=output)
        ixs = torch.argsort(scores, descending=True).tolist()
        return [data_manager.u_indices[i] for i in ixs[:num_annotate]]
