import numpy as np
import torch
from sklearn.datasets import make_regression

from pyrelational.datasets.base import BaseDataset

from .utils import create_splits


class SynthReg1(BaseDataset):
    """Synthetic dataset for active learning on a regression based task

    Simple 1 dof regression problem that can be placed into two types
    of AL situations as described in the module docstring

    :param n_splits: an int describing the number of class stratified
            splits to compute
    :param size: an int describing the number of observations the dataset
            is to have
    :param random_seed: random seed for reproducibility on splits
    """

    def __init__(self, n_splits: int = 5, size: int = 1000, random_seed: int = 1234):
        super(SynthReg1, self).__init__(n_splits=n_splits, random_seed=random_seed)
        self._create_data(size, random_seed)

    def _create_data(self, size: int, random_seed: int) -> None:
        x, y = make_regression(
            n_samples=size,
            n_features=1,
            n_targets=1,
            random_state=random_seed,
        )

        self.x = torch.FloatTensor(x)
        self.y = torch.FloatTensor(y)
        self.data_splits = create_splits(self.x, self.y, self.n_splits, self.random_seed)


class SynthReg2(BaseDataset):
    """Synthetic dataset for active learning on a regression based task

    A more challenging dataset than SynthReg1 wherein we see a periodic
    pattern with 2 degrees of freedom.

    :param n_splits: an int describing the number of class stratified
            splits to compute
    :param size: an int describing the number of observations the dataset
            is to have
    :param random_seed: random seed for reproducibility on splits
    """

    def __init__(self, n_splits: int = 5, size: int = 1000, random_seed: int = 1234):
        super(SynthReg2, self).__init__(n_splits=n_splits, random_seed=random_seed)
        self._create_data(size)

    def _create_data(self, size: int) -> None:
        zdata = 15 * np.random.random(size)
        xdata = np.sin(zdata) + 0.1 * np.random.randn(size)
        ydata = np.cos(zdata) + 0.1 * np.random.randn(size)

        zdata = torch.FloatTensor(zdata)
        xdata = torch.FloatTensor(xdata)
        ydata = torch.FloatTensor(ydata)

        self.x = torch.vstack([zdata, xdata]).T
        self.y = ydata
        self.data_splits = create_splits(self.x, self.y, self.n_splits, self.random_seed)
