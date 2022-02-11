from typing import List

import torch

from pyrelational.data import GenericDataManager
from pyrelational.informativeness import relative_distance
from pyrelational.models import GenericModel
from pyrelational.strategies.generic_al_strategy import GenericActiveLearningStrategy


class RelativeDistanceStrategy(GenericActiveLearningStrategy):
    """Diversity sampling based active learning strategy"""

    def __init__(
        self,
        data_manager: GenericDataManager,
        model: GenericModel,
    ):
        super(RelativeDistanceStrategy, self).__init__(data_manager, model)

    def active_learning_step(self, num_annotate: int, metric: str = "euclidean") -> List[int]:
        scores = relative_distance(self.u_loader, self.l_loader, metric=metric)
        ixs = torch.argsort(scores, descending=True).tolist()
        return [self.u_indices[i] for i in ixs[:num_annotate]]
