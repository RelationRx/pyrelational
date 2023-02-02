from torch import Tensor

from pyrelational.informativeness import regression_greedy_score
from pyrelational.strategies.regression.abstract_regression_strategy import (
    RegressionStrategy,
)


class MeanPredictionStrategy(RegressionStrategy):
    """Implements Greedy Strategy whereby unlabelled samples are queried based on their predicted mean value
    by the model"""

    def scoring_function(self, predictions: Tensor) -> Tensor:
        return regression_greedy_score(predictions)
