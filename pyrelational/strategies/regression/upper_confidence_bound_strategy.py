from typing import List

import torch

from pyrelational.data import GenericDataManager
from pyrelational.informativeness import regression_upper_confidence_bound
from pyrelational.models import GenericModel
from pyrelational.strategies.generic_al_strategy import GenericActiveLearningStrategy


class UpperConfidenceBoundStrategy(GenericActiveLearningStrategy):
    """Implements Upper Confidence Bound Strategy whereby unlabelled samples are scored and queried based on the
    UCB scorer"""

    def __init__(self, data_manager: GenericDataManager, model: GenericModel, kappa: float = 1.0):
        super(UpperConfidenceBoundStrategy, self).__init__(data_manager, model)
        self.kappa = kappa

    def active_learning_step(self, num_annotate: int) -> List[int]:
        self.model.train(self.l_loader, self.valid_loader)
        output = self.model(self.u_loader)
        uncertainty = regression_upper_confidence_bound(x=output, kappa=self.kappa)
        ixs = torch.argsort(uncertainty, descending=True).tolist()
        return [self.u_indices[i] for i in ixs[:num_annotate]]
