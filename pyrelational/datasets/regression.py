"""Regression datasets that can be used for benchmarking AL strategies
"""

import torch
import numpy as np
from torch.utils.data import Dataset 
from sklearn.datasets import make_regression, load_diabetes
from sklearn.model_selection import KFold
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

        X, y = make_regression(n_samples=size, 
            n_features=1, 
            n_targets=1, 
            random_state=random_seed)

        self.x = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

        kf = KFold(n_splits=n_splits)
        self.data_splits = kf.split(self.x, self.y)
        self.data_splits = [(idx[0], idx[1]) for idx in self.data_splits]

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]