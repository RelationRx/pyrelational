from pyrelational.data import DataManager
from pyrelational.informativeness import regression_least_confidence
from pyrelational.models import ModelManager
from pyrelational.strategies.regression.abstract_regression_strategy import (
    RegressionStrategy,
)


class LeastConfidenceStrategy(RegressionStrategy):
    """Implements Least Confidence Strategy whereby unlabelled samples are queried based on their predicted variance
    by the model"""

    def __init__(self):
        super(LeastConfidenceStrategy, self).__init__()
        self.scoring_fn = regression_least_confidence
