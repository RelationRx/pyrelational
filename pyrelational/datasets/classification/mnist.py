from typing import Tuple

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import ConcatDataset
from torchvision import datasets, transforms

from pyrelational.datasets.base import BaseDataset


class MNIST(BaseDataset):
    """
    MNIST dataset class that handles downloading, transforming, and loading MNIST data.

    :param data_dir: Directory to store or read the MNIST data.
    :param n_splits: Number of stratified splits for the dataset.
    :param random_seed: Seed for random number generator for reproducibility.
    """

    def __init__(self, data_dir: str = "/tmp/", random_seed: int = 1234):
        """Instantiate the MNIST dataset class.

        :param data_dir: directory where to download the data, defaults to "/tmp/"
        :param random_seed: random seed, defaults to 1234
        """
        super().__init__(random_seed=random_seed)
        self.data_dir = data_dir
        self._load_data()

    def _load_data(self) -> None:
        """Load the MNIST dataset from torchvision datasets.

        We apply the standard transformation with tensor conversation and normalisation.
        We concatenate the train and test datasets into a single dataset for unified handling, but
        we keep the same fixed test set.
        """
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        train_dataset = datasets.MNIST(root=self.data_dir, train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(root=self.data_dir, train=False, download=True, transform=transform)

        # Concatenate the train and test datasets
        self.full_dataset: ConcatDataset[Tuple[Tensor, Tensor]] = ConcatDataset([train_dataset, test_dataset])
        self.x = torch.stack([(self.full_dataset[i][0]) for i in range(len(self.full_dataset))])
        self.y = torch.stack([torch.tensor(self.full_dataset[i][1]) for i in range(len(self.full_dataset))])

        # Create splits for cross-validation
        train_ix, test_ix = np.arange(len(train_dataset)), np.arange(len(test_dataset)) + len(train_dataset)
        self.data_splits = [(train_ix, test_ix)]


if __name__ == "__main__":
    data = MNIST()
    print("plop")
