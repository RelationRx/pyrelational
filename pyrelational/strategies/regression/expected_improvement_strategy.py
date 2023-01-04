from typing import List

import torch

from pyrelational.data import DataManager
from pyrelational.informativeness import regression_expected_improvement
from pyrelational.models import ModelManager
from pyrelational.strategies.generic_al_strategy import GenericActiveLearningStrategy


class ExpectedImprovementStrategy(GenericActiveLearningStrategy):
    """Implement Expected Improvement Strategy whereby each unlabelled sample is scored based on the
    expected improvement scoring function. The top samples according to this score are selected at each step"""

    def __init__(self):
        super(ExpectedImprovementStrategy, self).__init__()

    def active_learning_step(self, num_annotate: int, data_manager: DataManager, model: ModelManager) -> List[int]:
        output = self.train_and_infer(data_manager=data_manager, model=model)
        max_label = max(data_manager.get_sample_labels(data_manager.l_indices))
        uncertainty = regression_expected_improvement(x=output, max_label=max_label)
        ixs = torch.argsort(uncertainty, descending=True).tolist()
        return [data_manager.u_indices[i] for i in ixs[:num_annotate]]
