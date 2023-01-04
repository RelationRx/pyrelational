from pyrelational.data import DataManager
from pyrelational.informativeness import regression_greedy_score
from pyrelational.models import ModelManager
from pyrelational.strategies.regression.generic_regression_strategy import (
    GenericRegressionStrategy,
)


class MeanPredictionStrategy(GenericRegressionStrategy):
    """Implements Greedy Strategy whereby unlabelled samples are queried based on their predicted mean value
    by the model"""

    def __init__(self):
        super(MeanPredictionStrategy, self).__init__()
        self.scoring_fn = regression_greedy_score
