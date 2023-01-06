from pyrelational.data import DataManager
from pyrelational.informativeness import regression_thompson_sampling
from pyrelational.models import ModelManager
from pyrelational.strategies.regression.generic_regression_strategy import (
    RegressionStrategy,
)


class ThompsonSamplingStrategy(RegressionStrategy):
    """Implements Thompson Sampling Strategy whereby unlabelled samples are scored and queried based on the
    thompson sampling scorer"""

    def __init__(self):
        super(ThompsonSamplingStrategy, self).__init__()
        self.scoring_fn = regression_thompson_sampling
