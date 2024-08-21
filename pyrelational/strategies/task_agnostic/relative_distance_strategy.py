"""Relative distance based active learning strategy."""

from pyrelational.informativeness import RelativeDistanceScorer
from pyrelational.samplers.samplers import DeterministicSampler
from pyrelational.strategies.abstract_strategy import Strategy


class RelativeDistanceStrategy(Strategy):
    """Diversity sampling based active learning strategy."""

    def __init__(self, metric: str = "euclidean"):
        """Initialise the strategy with a distance metric.

        :param metric: Name of distance metric to use. This should be supported by scikit-learn
            pairwise_distances function.
        """
        self.metric = metric
        super().__init__(RelativeDistanceScorer(metric=metric), DeterministicSampler())
