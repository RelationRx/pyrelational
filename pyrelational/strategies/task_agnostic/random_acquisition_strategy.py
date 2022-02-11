"""Defines and implements a random acquisition active learning strategy.
"""
from typing import List

import numpy as np

from pyrelational.data import GenericDataManager
from pyrelational.models import GenericModel
from pyrelational.strategies.generic_al_strategy import GenericActiveLearningStrategy


class RandomAcquisitionStrategy(GenericActiveLearningStrategy):
    """Implements RandomAcquisition whereby random samples from unlabelled set are chosen at each step"""

    def __init__(self, data_manager: GenericDataManager, model: GenericModel):
        super(RandomAcquisitionStrategy, self).__init__(data_manager, model)

    def active_learning_step(self, num_annotate: int) -> List[int]:
        num_annotate = min(num_annotate, len(self.u_indices))
        return np.random.choice(self.u_indices, size=num_annotate, replace=False).tolist()
