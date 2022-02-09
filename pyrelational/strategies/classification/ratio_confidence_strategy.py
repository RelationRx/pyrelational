"""
Active learning using ratio based confidence uncertainty measure
between classes in the posterior predictive distribution to
choose which observations to propose to the oracle
"""

from pyrelational.data.data_manager import GenericDataManager
from pyrelational.informativeness.classification import classification_ratio_confidence
from pyrelational.models.generic_model import GenericModel
from pyrelational.strategies.classification.generic_classification_strategy import (
    GenericClassificationStrategy,
)


class RatioConfidenceStrategy(GenericClassificationStrategy):
    """Implements Ratio Confidence Strategy whereby unlabelled samples are scored and queried based on
    the ratio confidence for classification scorer"""

    def __init__(self, data_manager: GenericDataManager, model: GenericModel):
        super(RatioConfidenceStrategy, self).__init__(data_manager, model)
        self.scoring_fn = classification_ratio_confidence
