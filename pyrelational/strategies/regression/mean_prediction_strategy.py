from torch import Tensor

from pyrelational.informativeness import regression_mean_prediction
from pyrelational.strategies.regression.abstract_regression_strategy import (
    RegressionStrategy,
)


class MeanPredictionStrategy(RegressionStrategy):
    """Implements Mean Prediction Strategy whereby unlabelled samples are queried based on their predicted mean value
    by the model. ie samples with the highest predicted mean values are queried."""

    def scoring_function(self, predictions: Tensor) -> Tensor:
        return regression_mean_prediction(predictions)
