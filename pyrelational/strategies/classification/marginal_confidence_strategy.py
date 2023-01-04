"""
Active learning using marginal confidence uncertainty measure
between classes in the posterior predictive distribution to
choose which observations to propose to the oracle
"""
from pyrelational.data import DataManager
from pyrelational.informativeness import classification_margin_confidence
from pyrelational.models import GenericModel
from pyrelational.strategies.classification.generic_classification_strategy import (
    GenericClassificationStrategy,
)


class MarginalConfidenceStrategy(GenericClassificationStrategy):
    """Implements Marginal Confidence Strategy whereby unlabelled samples are scored and queried based on
    the marginal confidence for classification scorer"""

    def __init__(self):
        super(MarginalConfidenceStrategy, self).__init__()
        self.scoring_fn = classification_margin_confidence
