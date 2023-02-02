"""
Active learning using entropy based confidence uncertainty measure
between classes in the posterior predictive distribution to
choose which observations to propose to the oracle
"""
from torch import Tensor

from pyrelational.informativeness import classification_entropy
from pyrelational.strategies.classification.abstract_classification_strategy import (
    ClassificationStrategy,
)


class EntropyClassificationStrategy(ClassificationStrategy):
    """Implements Entropy Classification Strategy whereby unlabelled samples are scored and queried based on
    entropy"""

    def scoring_function(self, predictions: Tensor) -> Tensor:
        return classification_entropy(predictions)
