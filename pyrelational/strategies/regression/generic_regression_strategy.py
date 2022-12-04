from abc import ABC

import torch

from pyrelational.data import GenericDataManager
from pyrelational.models import GenericModel
from pyrelational.strategies.generic_al_strategy import GenericActiveLearningStrategy


class GenericRegressionStrategy(GenericActiveLearningStrategy, ABC):
    """A base active learning strategy class for regression in which the top n indices,
    according to user-specified scoring function, are queried at each iteration"""

    def __init__(self):
        super(GenericRegressionStrategy, self).__init__()
        self.scoring_fn = NotImplementedError

    def active_learning_step(self, num_annotate: int, data_manager: GenericDataManager, model: GenericModel):
        model.train(data_manager.get_labelled_loader(), data_manager.get_validation_loader())
        output = model(data_manager.get_unlabelled_loader())
        scores = self.scoring_fn(x=output)
        ixs = torch.argsort(scores, descending=True).tolist()
        return [data_manager.u_indices[i] for i in ixs[:num_annotate]]
