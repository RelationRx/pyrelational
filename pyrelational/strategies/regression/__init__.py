"""Regression strategies for active learning."""

from pyrelational.strategies.regression.average_score_strategy import (
    AverageScoreStrategy,
)
from pyrelational.strategies.regression.bald_strategy import (
    BALDStrategy,
    SoftBALDStrategy,
)
from pyrelational.strategies.regression.expected_improvement_strategy import (
    ExpectedImprovementStrategy,
)
from pyrelational.strategies.regression.standard_deviation_strategy import (
    StandardDeviationStrategy,
)
from pyrelational.strategies.regression.thompson_sampling_strategy import (
    ThompsonSamplingStrategy,
)
from pyrelational.strategies.regression.upper_confidence_bound_strategy import (
    UpperConfidenceBoundStrategy,
)
