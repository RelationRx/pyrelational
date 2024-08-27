"""ClassificationStrategy class for active learning in classification tasks."""

import math
from typing import Any, List

import torch
from torch import Tensor

from pyrelational.data_managers import DataManager
from pyrelational.model_managers import ModelManager
from pyrelational.strategies.abstract_strategy import Strategy


class ClassificationStrategy(Strategy):
    """A base active learning strategy class for classification."""

    def __call__(
        self, num_annotate: int, data_manager: DataManager, model_manager: ModelManager[Any, Any]
    ) -> List[int]:
        """
        Identify samples for labelling based on user defined scoring and sampling function.

        :param num_annotate: number of samples to annotate
        :param data_manager: A pyrelational data manager
            which keeps track of what has been labelled and creates data loaders for
            active learning
        :param model_manager: A pyrelational model manager
            which wraps a user defined ML model to handle instantiation, training, testing,
            as well as uncertainty quantification

        :return: list of indices to annotate
        """
        output = self.train_and_infer(data_manager=data_manager, model_manager=model_manager).mean(0)
        if not torch.allclose(output.sum(1), torch.tensor(1.0)):
            output = softmax(output)
        uncertainty = self.scorer(output)
        return self.sampler(uncertainty, data_manager.u_indices, num_annotate)


def softmax(scores: Tensor, base: float = math.e, axis: int = -1) -> Tensor:
    """Return softmax array for array of scores.

    Converts a set of raw scores from a model (logits) into a
    probability distribution via softmax.

    The probability distribution will be a set of real numbers
    such that each is in the range 0-1.0 and the sum is 1.0.

    Assumes input is a pytorch tensor: tensor([1.0, 4.0, 2.0, 3.0])

    :param scores: (pytorch tensor) a pytorch tensor of any positive/negative real numbers.
    :param base: the base for the exponential (default e)
    :param: axis to apply softmax on scores

    :return: tensor of softmaxed scores
    """
    exps = base ** scores.float()  # exponential for each value in array
    sum_exps = torch.sum(exps, dim=axis, keepdim=True)  # sum of all exponentials
    prob_dist: Tensor = exps / sum_exps  # normalize exponentials
    return prob_dist
