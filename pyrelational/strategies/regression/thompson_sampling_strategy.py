from pyrelational.data.data_manager import GenericDataManager
from pyrelational.informativeness.regression import regression_thompson_sampling
from pyrelational.models.generic_model import GenericModel
from pyrelational.strategies.regression.generic_regression_strategy import (
    GenericRegressionStrategy,
)


class ThompsonSamplingStrategy(GenericRegressionStrategy):
    """Implements Thompson Sampling Strategy whereby unlabelled samples are scored and queried based on the
    thompson sampling scorer"""

    def __init__(self, data_manager: GenericDataManager, model: GenericModel):
        super(ThompsonSamplingStrategy, self).__init__(data_manager, model)
        self.scoring_fn = regression_thompson_sampling
