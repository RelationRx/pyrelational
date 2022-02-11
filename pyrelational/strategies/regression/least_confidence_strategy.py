from pyrelational.data import GenericDataManager
from pyrelational.informativeness import regression_least_confidence
from pyrelational.models import GenericModel
from pyrelational.strategies.regression.generic_regression_strategy import (
    GenericRegressionStrategy,
)


class LeastConfidenceStrategy(GenericRegressionStrategy):
    """Implements Least Confidence Strategy whereby unlabelled samples are queried based on their predicted variance
    by the model"""

    def __init__(self, data_manager: GenericDataManager, model: GenericModel):
        super(LeastConfidenceStrategy, self).__init__(data_manager, model)
        self.scoring_fn = regression_least_confidence
