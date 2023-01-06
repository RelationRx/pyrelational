from abc import ABC

import torch

from pyrelational.data import DataManager
from pyrelational.informativeness import softmax
from pyrelational.models import ModelManager
from pyrelational.strategies.generic_al_strategy import Strategy


class ClassificationStrategy(Strategy, ABC):
    """A base active learning strategy class for classification in which the top n indices,
    according to user-specified scoring function, are queried at each iteration"""

    def __init__(self):
        super(ClassificationStrategy, self).__init__()
        self.scoring_fn = NotImplementedError

    def active_learning_step(self, num_annotate: int, data_manager: DataManager, model: ModelManager):
        output = self.train_and_infer(data_manager=data_manager, model=model).mean(0)
        if not torch.allclose(output.sum(1), torch.tensor(1.0)):
            output = softmax(output)
        uncertainty = self.scoring_fn(softmax(output))
        ixs = torch.argsort(uncertainty, descending=True).tolist()
        return [data_manager.u_indices[i] for i in ixs[:num_annotate]]
