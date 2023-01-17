from typing import List

import torch

from pyrelational.data import DataManager
from pyrelational.informativeness import regression_upper_confidence_bound
from pyrelational.models import ModelManager
from pyrelational.strategies.abstract_strategy import Strategy


class UpperConfidenceBoundStrategy(Strategy):
    """Implements Upper Confidence Bound Strategy whereby unlabelled samples are scored and queried based on the
    UCB scorer"""

    def __init__(self, kappa: float = 1.0):
        super(UpperConfidenceBoundStrategy, self).__init__()
        self.kappa = kappa

    def __call__(self, num_annotate: int, data_manager: DataManager, model: ModelManager) -> List[int]:
        output = self.train_and_infer(data_manager=data_manager, model=model)
        uncertainty = regression_upper_confidence_bound(x=output.squeeze(-1), kappa=self.kappa)
        ixs = torch.argsort(uncertainty, descending=True).tolist()
        return [data_manager.u_indices[i] for i in ixs[:num_annotate]]
