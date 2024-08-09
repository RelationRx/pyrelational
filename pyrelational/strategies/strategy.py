"""Flexible active learning strategy."""

from typing import Any, List

from pyrelational.data_managers import DataManager
from pyrelational.informativeness.abstract_scorers import AbstractScorer
from pyrelational.model_managers import ModelManager
from pyrelational.samplers.abstract_sampler import AbstractSampler
from pyrelational.strategies.abstract_strategy import Strategy


class GenericStrategy(Strategy):
    """This module can be used to define flexible active learning strategy.

    This relies on the assumption that (most) active learning strategies can be
    decomposed into two components: a scoring function and a sampler. Thus to initialize
    this class, simply pass a scorer and a sampler.
    """

    def __init__(self, scorer: AbstractScorer, sampler: AbstractSampler):
        """Initialize the strategy with a scorer and a sampler.

        :param scorer: instance of a scorer class
        :param sampler: instance of a sampler class
        """
        self.scorer = scorer
        self.sampler = sampler

    def __call__(
        self, num_annotate: int, data_manager: DataManager, model_manager: ModelManager[Any, Any]
    ) -> List[int]:
        """Call function which identifies samples which need to be labelled based on user defined scoring function.

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
