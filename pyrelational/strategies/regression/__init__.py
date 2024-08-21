"""Regression strategies for active learning."""

from pyrelational.strategies.regression.bald_strategy import (
    BALDStrategy,
    SoftBALDStrategy,
)
from pyrelational.strategies.regression.expected_improvement_strategy import (
    ExpectedImprovementStrategy,
)
from pyrelational.strategies.regression.greedy_strategy import GreedyStrategy
from pyrelational.strategies.regression.regression_strategy import RegressionStrategy
from pyrelational.strategies.regression.thompson_sampling_strategy import (
    ThompsonSamplingStrategy,
)
from pyrelational.strategies.regression.upper_confidence_bound_strategy import (
    UpperConfidenceBoundStrategy,
)
from pyrelational.strategies.regression.variance_reduction_strategy import (
    VarianceReductionStrategy,
)
