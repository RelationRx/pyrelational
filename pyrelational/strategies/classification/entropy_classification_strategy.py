"""
Active learning using entropy based confidence uncertainty measure
between classes in the posterior predictive distribution to
choose which observations to propose to the oracle
"""

from pyrelational.data import DataManager
from pyrelational.informativeness import classification_entropy
from pyrelational.models import ModelManager
from pyrelational.strategies.classification.generic_classification_strategy import (
    ClassificationStrategy,
)


class EntropyClassificationStrategy(ClassificationStrategy):
    """Implements Entropy Classification Strategy whereby unlabelled samples are scored and queried based on
    entropy"""

    def __init__(self):
        super(EntropyClassificationStrategy, self).__init__()
        self.scoring_fn = classification_entropy
