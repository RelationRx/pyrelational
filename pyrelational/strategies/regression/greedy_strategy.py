from pyrelational.data.data_manager import GenericDataManager
from pyrelational.informativeness.regression import regression_greedy_score
from pyrelational.models.generic_model import GenericModel
from pyrelational.strategies.regression.generic_regression_strategy import (
    GenericRegressionStrategy,
)


class GreedyStrategy(GenericRegressionStrategy):
    """Implements Greedy Strategy whereby unlabelled samples are queried based on their predicted mean value
    by the model"""

    def __init__(self, data_manager: GenericDataManager, model: GenericModel):
        super(GreedyStrategy, self).__init__(data_manager, model)
        self.scoring_fn = regression_greedy_score
