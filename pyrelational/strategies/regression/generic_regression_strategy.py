from abc import ABC

import torch

from pyrelational.data import GenericDataManager
from pyrelational.models import GenericModel
from pyrelational.strategies.generic_al_strategy import GenericActiveLearningStrategy


class GenericRegressionStrategy(GenericActiveLearningStrategy, ABC):
    """A base active learning strategy class for regression in which the top n indices,
    according to user-specified scoring function, are queried at each iteration"""

    def __init__(self, data_manager: GenericDataManager, model: GenericModel):
        super(GenericRegressionStrategy, self).__init__(data_manager, model)
        self.scoring_fn = NotImplementedError

    def active_learning_step(self, num_annotate: int):
        self.model.train(self.l_loader, self.valid_loader)
        output = self.model(self.u_loader)
        scores = self.scoring_fn(x=output)
        ixs = torch.argsort(scores, descending=True).tolist()
        return [self.u_indices[i] for i in ixs[:num_annotate]]
