"""Least Confidence Strategy for regression tasks."""

from pyrelational.batch_mode_samplers import TopKSampler
from pyrelational.informativeness import StandardDeviation
from pyrelational.strategies.regression.regression_strategy import RegressionStrategy


class VarianceReductionStrategy(RegressionStrategy):
    """Implements Least Confidence Strategy.

    Unlabelled samples are queried based on their predicted variance by the model.
    """

    def __init__(self, axis: int = 0):
        """Initialize the strategy with the least confidence scorer and a deterministic scorer for regression."""
        super().__init__(StandardDeviation(axis=axis), TopKSampler())
