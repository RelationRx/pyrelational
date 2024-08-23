"""Greedy Strategy Module."""

from pyrelational.batch_mode_samplers import TopKSampler
from pyrelational.informativeness import AverageScorer
from pyrelational.strategies.regression.regression_strategy import RegressionStrategy


class GreedyStrategy(RegressionStrategy):
    """Implements Greedy Strategy.

    Unlabelled samples are queried based on their predicted mean value by the model.
    ie samples with the highest predicted mean values are queried.
    """

    def __init__(self, axis: int = 0):
        """Initialize the strategy with the mean prediction scorer and a deterministic scorer for regression."""
        super().__init__(AverageScorer(axis=axis), TopKSampler())
