from typing import List

import torch

from pyrelational.data import GenericDataManager
from pyrelational.informativeness import regression_upper_confidence_bound
from pyrelational.models import GenericModel
from pyrelational.strategies.generic_al_strategy import GenericActiveLearningStrategy


class UpperConfidenceBoundStrategy(GenericActiveLearningStrategy):
    """Implements Upper Confidence Bound Strategy whereby unlabelled samples are scored and queried based on the
    UCB scorer"""

    def __init__(self, kappa: float = 1.0):
        super(UpperConfidenceBoundStrategy, self).__init__()
        self.kappa = kappa

    def active_learning_step(
        self, num_annotate: int, data_manager: GenericDataManager, model: GenericModel
    ) -> List[int]:
        model.train(data_manager.get_labelled_loader(), data_manager.get_validation_loader())
        output = model(data_manager.get_unlabelled_loader())
        uncertainty = regression_upper_confidence_bound(x=output, kappa=self.kappa)
        ixs = torch.argsort(uncertainty, descending=True).tolist()
        return [data_manager.u_indices[i] for i in ixs[:num_annotate]]
