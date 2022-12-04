from typing import List

import torch

from pyrelational.data import GenericDataManager
from pyrelational.informativeness import relative_distance
from pyrelational.strategies.generic_al_strategy import GenericActiveLearningStrategy


class RelativeDistanceStrategy(GenericActiveLearningStrategy):
    """Diversity sampling based active learning strategy"""

    def __init__(self):
        super(RelativeDistanceStrategy, self).__init__()

    def active_learning_step(
        self, num_annotate: int, data_manager: GenericDataManager, metric: str = "euclidean"
    ) -> List[int]:
        scores = relative_distance(
            data_manager.get_unlabelled_loader(), data_manager.get_labelled_loader(), metric=metric
        )
        ixs = torch.argsort(scores, descending=True).tolist()
        return [data_manager.u_indices[i] for i in ixs[:num_annotate]]
