"""
Active learning using ratio based confidence uncertainty measure
between classes in the posterior predictive distribution to
choose which observations to propose to the oracle
"""
from pyrelational.informativeness import classification_ratio_confidence
from pyrelational.strategies.classification.generic_classification_strategy import (
    ClassificationStrategy,
)


class RatioConfidenceStrategy(ClassificationStrategy):
    """Implements Ratio Confidence Strategy whereby unlabelled samples are scored and queried based on
    the ratio confidence for classification scorer"""

    def __init__(self):
        super(RatioConfidenceStrategy, self).__init__()
        self.scoring_fn = classification_ratio_confidence
