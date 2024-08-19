"""Thomas Sampling Strategy for Regression."""

from pyrelational.informativeness import ThompsonSampling
from pyrelational.samplers.samplers import DeterministicSampler
from pyrelational.strategies.regression.regression_strategy import RegressionStrategy


class ThompsonSamplingStrategy(RegressionStrategy):
    """Implements Thompson Sampling Strategy.

    Unlabelled samples are scored and queried based on the thompson sampling scorer.
    """

    def __init__(self, axis: int = 0):
        """Initialize the strategy with the thompson sampling scorer and a deterministic scorer for regression."""
        super().__init__(ThompsonSampling(axis=axis), DeterministicSampler())
