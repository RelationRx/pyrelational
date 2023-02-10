"""Defines and implements a random acquisition active learning strategy.
"""
from typing import List

import numpy as np

from pyrelational.data_managers import DataManager
from pyrelational.strategies.abstract_strategy import Strategy


class RandomAcquisitionStrategy(Strategy):
    """Implements RandomAcquisition whereby random samples from unlabelled set are chosen at each step"""

    def __init__(self) -> None:
        super(RandomAcquisitionStrategy, self).__init__()

    def __call__(self, num_annotate: int, data_manager: DataManager) -> List[int]:
        """
        Call function which identifies samples which need to be labelled

        :param num_annotate: number of samples to annotate
        :param data_manager: A pyrelational data manager
            which keeps track of what has been labelled and creates data loaders for
            active learning

        :return: list of indices to annotate
        """
        num_annotate = min(num_annotate, len(data_manager.u_indices))
        ret: List[int] = np.random.choice(data_manager.u_indices, size=num_annotate, replace=False).tolist()
        return ret
