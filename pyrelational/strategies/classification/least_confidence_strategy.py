"""
Active learning using least confidence uncertainty measure
between classes in the posterior predictive distribution to
choose which observations to propose to the oracle
"""

from pyrelational.data import GenericDataManager
from pyrelational.informativeness import classification_least_confidence
from pyrelational.models import GenericModel
from pyrelational.strategies.classification.generic_classification_strategy import (
    GenericClassificationStrategy,
)


class LeastConfidenceStrategy(GenericClassificationStrategy):
    """Implements Least Confidence Strategy whereby unlabelled samples are scored and queried based on
    the least confidence for classification scorer"""

    def __init__(self, data_manager: GenericDataManager, model: GenericModel):
        super(LeastConfidenceStrategy, self).__init__(data_manager, model)
        self.scoring_fn = classification_least_confidence
