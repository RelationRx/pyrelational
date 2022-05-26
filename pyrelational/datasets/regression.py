"""Regression datasets that can be used for benchmarking AL strategies
"""

import numpy as np
import torch
from sklearn.datasets import load_diabetes, make_regression
from sklearn.model_selection import KFold
from torch.utils.data import Dataset

from .uci_datasets import UCIDatasets


class SynthReg1(Dataset):
    """Synthetic dataset for active learning on a regression based task

    Simple 1 dof regression problem that can be placed into two types
    of AL situations as described in the module docstring

    :param n_splits: an int describing the number of class stratified
            splits to compute
    :param size: an int describing the number of observations the dataset
            is to have
    :param random_seed: random seed for reproducibility on splits
    """

    def __init__(self, n_splits=5, size=1000, random_seed=1234):
        super(SynthReg1, self).__init__()
        self.size = size
        self.random_seed = random_seed
        self.n_splits = n_splits

        X, y = make_regression(n_samples=size, n_features=1, n_targets=1, random_state=random_seed)

        self.x = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

        kf = KFold(n_splits=n_splits)
        self.data_splits = kf.split(self.x, self.y)
        self.data_splits = [(idx[0], idx[1]) for idx in self.data_splits]

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class SynthReg2(Dataset):
    """Synthetic dataset for active learning on a regression based task

    A more challenging dataset than SynthReg1 wherein we see a periodic
    pattern with 2 degrees of freedom.

    :param n_splits: an int describing the number of class stratified
            splits to compute
    :param size: an int describing the number of observations the dataset
            is to have
    :param random_seed: random seed for reproducibility on splits
    """

    def __init__(self, n_splits=5, size=1000, random_seed=1234):
        super(SynthReg2, self).__init__()
        self.size = size
        self.random_seed = random_seed
        self.n_splits = n_splits

        # Samples
        zdata = 15 * np.random.random(size)
        xdata = np.sin(zdata) + 0.1 * np.random.randn(size)
        ydata = np.cos(zdata) + 0.1 * np.random.randn(size)

        # Convert
        zdata = torch.FloatTensor(zdata)
        xdata = torch.FloatTensor(xdata)
        ydata = torch.FloatTensor(ydata)

        self.x = torch.vstack([zdata, xdata]).T
        self.y = ydata

        kf = KFold(n_splits=n_splits)
        self.data_splits = kf.split(self.x, self.y)
        self.data_splits = [(idx[0], idx[1]) for idx in self.data_splits]

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class DiabetesDataset(Dataset):
    """A small regression dataset for examples

    From Bradley Efron, Trevor Hastie, Iain Johnstone and
    Robert Tibshirani (2004) “Least Angle Regression,”
    Annals of Statistics (with discussion), 407-499.

    :param n_splits: an int describing the number of class stratified
        splits to compute
    """

    def __init__(self, n_splits=5):
        # Load the diabetes dataset
        diabetes_X, diabetes_y = load_diabetes(return_X_y=True)
        self.x = torch.FloatTensor(diabetes_X)
        self.y = torch.FloatTensor(diabetes_y)

        kf = KFold(n_splits=n_splits)
        self.data_splits = kf.split(self.x, self.y)
        self.data_splits = [(idx[0], idx[1]) for idx in self.data_splits]

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class UCIRegression(Dataset):
    """UCI regression dataset base class

    :param n_splits: an int describing the number of class stratified
        splits to compute
    """

    def __init__(self, name, data_dir="/tmp/", n_splits=5):
        super(UCIRegression, self).__init__()
        dataset = UCIDatasets(name=name, data_dir=data_dir, n_splits=n_splits)
        self.data_dir = dataset.data_dir
        self.name = dataset.name
        self.data_splits = dataset.data_splits

        dataset = dataset.get_simple_dataset()
        self.len_dataset = len(dataset)
        self.x = dataset[:][0]
        self.y = dataset[:][1].squeeze()

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class UCIConcrete(UCIRegression):
    """UCI housing dataset

    :param n_splits: an int describing the number of class stratified
        splits to compute
    """

    def __init__(self, data_dir="/tmp/", n_splits=5):
        super(UCIConcrete, self).__init__(name="concrete", data_dir=data_dir, n_splits=n_splits)


class UCIEnergy(UCIRegression):
    """UCI housing dataset

    :param n_splits: an int describing the number of class stratified
        splits to compute
    """

    def __init__(self, data_dir="/tmp/", n_splits=5):
        super(UCIEnergy, self).__init__(name="energy", data_dir=data_dir, n_splits=n_splits)


class UCIPower(UCIRegression):
    """UCI housing dataset

    :param n_splits: an int describing the number of class stratified
        splits to compute
    """

    def __init__(self, data_dir="/tmp/", n_splits=5):
        super(UCIPower, self).__init__(name="power", data_dir=data_dir, n_splits=n_splits)


class UCIWine(UCIRegression):
    """UCI housing dataset

    :param n_splits: an int describing the number of class stratified
        splits to compute
    """

    def __init__(self, data_dir="/tmp/", n_splits=5):
        super(UCIWine, self).__init__(name="wine", data_dir=data_dir, n_splits=n_splits)


class UCIYacht(UCIRegression):
    """UCI housing dataset

    :param n_splits: an int describing the number of class stratified
        splits to compute
    """

    def __init__(self, data_dir="/tmp/", n_splits=5):
        super(UCIYacht, self).__init__(name="yacht", data_dir=data_dir, n_splits=n_splits)


class UCIAirfoil(UCIRegression):
    """UCI Airfoil dataset

    :param n_splits: an int describing the number of class stratified
        splits to compute
    """

    def __init__(self, data_dir="/tmp/", n_splits=5):
        super(UCIAirfoil, self).__init__(name="airfoil", data_dir=data_dir, n_splits=n_splits)
