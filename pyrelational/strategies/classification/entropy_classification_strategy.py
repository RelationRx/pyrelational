"""
Active learning using entropy based confidence uncertainty measure
between classes in the posterior predictive distribution to
choose which observations to propose to the oracle
"""

from pyrelational.data import GenericDataManager
from pyrelational.informativeness import classification_entropy
from pyrelational.models import GenericModel
from pyrelational.strategies.classification.generic_classification_strategy import (
    GenericClassificationStrategy,
)


class EntropyClassificationStrategy(GenericClassificationStrategy):
    """Implements Entropy Classification Strategy whereby unlabelled samples are scored and queried based on
    entropy"""

    def __init__(self, data_manager: GenericDataManager, model: GenericModel):
        super(EntropyClassificationStrategy, self).__init__(data_manager, model)
        self.scoring_fn = classification_entropy
