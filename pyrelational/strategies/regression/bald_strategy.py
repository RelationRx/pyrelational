"""BALD Strategy for regression tasks."""

from pyrelational.batch_mode_samplers import ProbabilisticSampler, TopKSampler
from pyrelational.informativeness import RegressionBald
from pyrelational.strategies.regression.regression_strategy import RegressionStrategy


class BALDStrategy(RegressionStrategy):
    """Implements BALD Strategy.

    Samples are queried based on mutual information score based on multiple estimator models.
    """

    def __init__(self, axis: int = 0):
        """Initialise the strategy with bald scorer and deterministic sampler."""
        super().__init__(RegressionBald(axis=axis), TopKSampler())


class SoftBALDStrategy(RegressionStrategy):
    """Implements soft BALD Strategy.

    Unlabelled samples are queried based on mutual information score based on
    multiple estimator models. In contrast to Bald the query is drawn from unlabelled pool based on probabilities
    derived from bald scores instead of using an argmax operation.
    """

    def __init__(self, axis: int = 0):
        """Initialise the strategy with bald scorer and probabilistic sampler."""
        super().__init__(RegressionBald(axis=axis), ProbabilisticSampler())
