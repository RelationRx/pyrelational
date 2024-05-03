import numpy as np
import torch

from pyrelational.datasets.base import BaseDataset
from .utils import create_splits


class AIAUSynth(BaseDataset):
    """
    2D example dataset used in Scherer et al. 2024 for active learning

    This is a 2D regression problem with a single output. The function is
    g(x1, x2) = sin(1.5 * x1) * sin(1.5 * x2). To replicate the results in
    the paper one will have to compute the evaluation over training and test,
    as it attempts to study how quickly the model can learn the function over
    the full input space. This is only applicable to the "T1" scenario
    described in the paper.

    :param n_splits: an int describing the number of class stratified
            splits to compute
    :param random_seed: random seed for reproducibility on splits
    """

    def __init__(self, n_splits: int = 5, random_seed: int = 1234):
        super().__init__(n_splits=n_splits, random_seed=random_seed)
        self._create_data(random_seed)

    def _create_data(self, random_seed: int) -> None:

        # define the range of x1 and x2 values
        x1_range = np.linspace(0, 2 * np.pi, 50)
        x2_range = np.linspace(0, 2 * np.pi, 50)

        # Create a grid of x1 and x2 values
        x1, x2 = np.meshgrid(x1_range, x2_range)

        # Calculate the function values
        g = np.sin(1.5 * x1) * np.sin(1.5 * x2)  # no noise

        # construct x as two dimensional vector observations
        x1 = x1.flatten()
        x2 = x2.flatten()
        self.x = np.c_[x1, x2]
        self.y = g.flatten()

        self.x = torch.FloatTensor(self.x)
        self.y = torch.FloatTensor(self.y)
        self.data_splits = create_splits(self.x, self.y, self.n_splits, self.random_seed)
