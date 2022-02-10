from typing import List

import torch

from pyrelational.data import GenericDataManager
from pyrelational.informativeness import regression_expected_improvement
from pyrelational.models import GenericModel
from pyrelational.strategies.generic_al_strategy import GenericActiveLearningStrategy


class ExpectedImprovementStrategy(GenericActiveLearningStrategy):
    """Implement Expected Improvement Strategy whereby each unlabelled sample is scored based on the
    expected improvement scoring function. The top samples according to this score are selected at each step"""

    def __init__(self, data_manager: GenericDataManager, model: GenericModel):
        super(ExpectedImprovementStrategy, self).__init__(data_manager, model)

    def active_learning_step(self, num_annotate: int) -> List[int]:
        self.model.train(self.l_loader, self.valid_loader)
        output = self.model(self.u_loader)
        max_label = max(self.data_manager.get_sample_labels(self.l_indices))
        uncertainty = regression_expected_improvement(x=output, max_label=max_label)
        ixs = torch.argsort(uncertainty, descending=True).tolist()
        return [self.u_indices[i] for i in ixs[:num_annotate]]
