"""Relative distance based active learning strategy."""

from typing import List

from pyrelational.batch_mode_samplers import TopKSampler
from pyrelational.data_managers import DataManager
from pyrelational.informativeness import RelativeDistanceScorer
from pyrelational.strategies.abstract_strategy import Strategy


class RelativeDistanceStrategy(Strategy):
    """Diversity sampling based active learning strategy."""

    scorer: RelativeDistanceScorer

    def __init__(self, metric: str = "euclidean"):
        """Initialise the strategy with a distance metric.

        :param metric: Name of distance metric to use. This should be supported by scikit-learn
            pairwise_distances function.
        """
        self.metric = metric
        super().__init__(RelativeDistanceScorer(metric=metric), TopKSampler())

    def __call__(self, num_annotate: int, data_manager: DataManager) -> List[int]:
        """Identify samples which need to be labelled.

        :param num_annotate: number of samples to annotate
        :param data_manager: A pyrelational data manager
            which keeps track of what has been labelled and creates data loaders for
            active learning

        :return: list of indices to annotate
        """
        scores = self.scorer(data_manager.get_unlabelled_loader(), data_manager.get_labelled_loader())
        return self.sampler(scores, data_manager.u_indices, num_annotate)
