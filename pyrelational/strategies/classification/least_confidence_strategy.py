"""
Active learning using least confidence uncertainty measure
between classes in the posterior predictive distribution to
choose which observations to propose to the oracle
"""
from torch import Tensor

from pyrelational.informativeness import classification_least_confidence
from pyrelational.strategies.classification.abstract_classification_strategy import (
    ClassificationStrategy,
)


class LeastConfidenceStrategy(ClassificationStrategy):
    """Implements Least Confidence Strategy whereby unlabelled samples are scored and queried based on
    the least confidence for classification scorer"""

    def scoring_function(self, predictions: Tensor) -> Tensor:
        return classification_least_confidence(predictions)
