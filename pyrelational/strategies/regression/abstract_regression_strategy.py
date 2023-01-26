from abc import ABC
from typing import List

import torch

from pyrelational.data import DataManager
from pyrelational.models import ModelManager
from pyrelational.strategies.abstract_strategy import Strategy


class RegressionStrategy(Strategy, ABC):
    """A base active learning strategy class for regression in which the top n indices,
    according to user-specified scoring function, are queried at each iteration"""

    def __init__(self):
        super(RegressionStrategy, self).__init__()
        self.scoring_fn = NotImplementedError

    def __call__(self, num_annotate: int, data_manager: DataManager, model: ModelManager) -> List[int]:
        """
        Call function which identifies samples which need to be labelled

        :param num_annotate: number of samples to annotate
        :param data_manager: A pyrelational data manager
            which keeps track of what has been labelled and creates data loaders for
            active learning
        :param model: A pyrelational model manager
            which wraps a user defined ML model to handle instantiation, training, testing,
            as well as uncertainty quantification

        :return: list of indices to annotate
        """
        output = self.train_and_infer(data_manager=data_manager, model=model)
        scores = self.scoring_fn(x=output)
        ixs = torch.argsort(scores, descending=True).tolist()
        return [data_manager.u_indices[i] for i in ixs[:num_annotate]]
