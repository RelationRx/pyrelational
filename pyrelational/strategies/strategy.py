from typing import Any, List

from pyrelational.data_managers import DataManager
from pyrelational.informativeness.abstract_scorers import AbstractScorer
from pyrelational.model_managers import ModelManager
from pyrelational.samplers.abstract_sampler import AbstractSampler
from pyrelational.strategies.abstract_strategy import Strategy


class GenericStrategy(Strategy):
    """
    This module defines an abstract active learning strategy.

    Any strategy should be a subclass of this class and override the `__call__` method to suggest observations
    to be labeled. In the general case `__call__` would be the composition of an informativeness function,
    which assigns a measure of informativeness to unlabelled observations, and a selection algorithm which
    chooses what observations to present to the oracle.

    The user defined __call__ method must have a "num_annotate" argument
    """

    def __init__(self, scorer: AbstractScorer, sampler: AbstractSampler):
        self.scorer = scorer
        self.sampler = sampler

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
        scores = self.scorer(output)
        return self.sampler(scores, data_manager.u_indices, num_annotate)
