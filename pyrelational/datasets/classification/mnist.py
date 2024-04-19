from typing import Tuple

import torch
from torch import Tensor
from torch.utils.data import ConcatDataset
from torchvision import datasets, transforms

from .base import BaseDataset


class FashionMNIST(BaseDataset):
    """
    Fashion MNIST dataset class that handles downloading, transforming, and loading Fashion MNIST data.
    This dataset includes images from 10 categories of clothing, each represented as a 28x28 grayscale image.

    :param data_dir: Directory to store or read the Fashion MNIST data.
    :param n_splits: Number of stratified splits for the dataset.
    :param random_seed: Seed for random number generator for reproducibility.
    """

    def __init__(self, data_dir: str = "/tmp/", n_splits: int = 5, random_seed: int = 1234):
        super().__init__(n_splits=n_splits, random_seed=random_seed)
        self.data_dir = data_dir
        self._load_data()

    def _load_data(self) -> None:
        """
        Loads the Fashion MNIST dataset from torchvision datasets, applies a transformation
        to convert images to tensors, and concatenates the train and test datasets into
        a single dataset for unified handling.
        """
        # Transform to convert images to PyTorch tensors
        transform = transforms.Compose(
            [
                transforms.ToTensor(),  # Transform the image to a PyTorch tensor
                transforms.Normalize((0.5,), (0.5,)),  # Normalize the images to [-1, 1]
            ]
        )

        # Download the training and test datasets
        train_dataset = datasets.FashionMNIST(root=self.data_dir, train=True, download=True, transform=transform)
        test_dataset = datasets.FashionMNIST(root=self.data_dir, train=False, download=True, transform=transform)

        # Concatenate the train and test datasets
        self.full_dataset: ConcatDataset[Tuple[Tensor, Tensor]] = ConcatDataset([train_dataset, test_dataset])
        self.x = torch.stack([(self.full_dataset[i][0]).flatten() for i in range(len(self.full_dataset))])
        self.y = torch.stack([torch.tensor(self.full_dataset[i][1]) for i in range(len(self.full_dataset))])

        # Create splits for cross-validation
        self._create_splits()
