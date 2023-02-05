from torch import Tensor

from pyrelational.informativeness import regression_thompson_sampling
from pyrelational.strategies.regression.abstract_regression_strategy import (
    RegressionStrategy,
)


class ThompsonSamplingStrategy(RegressionStrategy):
    """Implements Thompson Sampling Strategy whereby unlabelled samples are scored and queried based on the
    thompson sampling scorer"""

    def scoring_function(self, predictions: Tensor) -> Tensor:
        return regression_thompson_sampling(predictions)
