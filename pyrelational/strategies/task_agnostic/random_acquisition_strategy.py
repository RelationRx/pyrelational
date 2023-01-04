"""Defines and implements a random acquisition active learning strategy.
"""
from typing import List

import numpy as np

from pyrelational.data import DataManager
from pyrelational.strategies.generic_al_strategy import Strategy


class RandomAcquisitionStrategy(Strategy):
    """Implements RandomAcquisition whereby random samples from unlabelled set are chosen at each step"""

    def __init__(self):
        super(RandomAcquisitionStrategy, self).__init__()

    def active_learning_step(self, num_annotate: int, data_manager: DataManager) -> List[int]:
        num_annotate = min(num_annotate, len(data_manager.u_indices))
        return np.random.choice(data_manager.u_indices, size=num_annotate, replace=False).tolist()
