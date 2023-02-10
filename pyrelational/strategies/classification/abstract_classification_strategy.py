from abc import ABC, abstractmethod
from typing import Any, List

import torch
from torch import Tensor

from pyrelational.data_managers import DataManager
from pyrelational.informativeness import softmax
from pyrelational.model_managers import ModelManager
from pyrelational.strategies.abstract_strategy import Strategy


class ClassificationStrategy(Strategy, ABC):
    """
    A base active learning strategy class for classification in which the top n indices,
    according to user-specified scoring function, are queried at each iteration.
    """

    def __init__(self) -> None:
        super(ClassificationStrategy, self).__init__()

    @abstractmethod
    def scoring_function(self, predictions: Tensor) -> Tensor:
        """
        Compute score of each sample.

        :param predictions: model predictions for each sample
        :return: scores for each sample
        """

    def __call__(
        self, num_annotate: int, data_manager: DataManager, model_manager: ModelManager[Any, Any]
    ) -> List[int]:
        """
        Call function which identifies samples which need to be labelled based on
        user defined scoring function.

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
        uncertainty = self.scoring_function(softmax(output))
        ixs = torch.argsort(uncertainty, descending=True).tolist()
        return [data_manager.u_indices[i] for i in ixs[:num_annotate]]
