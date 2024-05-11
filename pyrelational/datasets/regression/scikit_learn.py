import torch
from sklearn.datasets import load_diabetes

from pyrelational.datasets.base import BaseDataset

from .utils import create_splits


class DiabetesDataset(BaseDataset):
    """A small regression dataset for examples

    From Bradley Efron, Trevor Hastie, Iain Johnstone and
    Robert Tibshirani (2004) “Least Angle Regression,”
    Annals of Statistics (with discussion), 407-499.

    :param n_splits: an int describing the number of class stratified
        splits to compute
    """

    def __init__(self, n_splits: int = 5, random_seed: int = 0):
        super().__init__(n_splits=n_splits, random_seed=random_seed)
        # Load the diabetes dataset
        diabetes_X, diabetes_y = load_diabetes(return_X_y=True)
        self.x = torch.FloatTensor(diabetes_X)
        self.y = torch.FloatTensor(diabetes_y)

        self.data_splits = create_splits(self.x, self.y, self.n_splits, self.random_seed)
