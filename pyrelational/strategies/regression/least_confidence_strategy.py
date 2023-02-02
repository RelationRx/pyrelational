from torch import Tensor

from pyrelational.informativeness import regression_least_confidence
from pyrelational.strategies.regression.abstract_regression_strategy import (
    RegressionStrategy,
)


class LeastConfidenceStrategy(RegressionStrategy):
    """Implements Least Confidence Strategy whereby unlabelled samples are queried based on their predicted variance
    by the model"""

    def scoring_function(self, predictions: Tensor) -> Tensor:
        return regression_least_confidence(predictions)
