"""Upper Confidence Bound Strategy."""

from pyrelational.batch_mode_samplers import TopKSampler
from pyrelational.informativeness import UpperConfidenceBound
from pyrelational.strategies.regression.regression_strategy import RegressionStrategy


class UpperConfidenceBoundStrategy(RegressionStrategy):
    """Implements Upper Confidence Bound Strategy.

    Unlabelled samples are scored and queried based on the UCB scorer.
    """

    def __init__(self, kappa: float = 1.0, axis: int = 0):
        """Initialize the strategy with the UCB scorer and a deterministic scorer for regression.

        :param kappa: trade-off parameter between exploitation and exploration
        """
        super().__init__(UpperConfidenceBound(kappa=kappa, axis=axis), TopKSampler())
