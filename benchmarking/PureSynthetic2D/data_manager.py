"""
Dataset and DataManager for the 2D synthetic regression task

License: MIT

Experimental setup: B

The dataset is generated using the following function:
"""

from typing import Any, List, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset

from pyrelational.data_managers import DataManager


def numpy_collate(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Any:
    """Collate function for a Pytorch to Numpy DataLoader"""
    x_list, y_list = zip(*batch)
    return [np.stack(x_list), np.stack(y_list)]


class Synthetic2D(Dataset[Tuple[torch.Tensor, torch.Tensor]]):
    """
    Dataset implementation for the 2D synthetic regression task

    Args:
        x1_size (int): Number of samples in the first dimension
        x2_size (int): Number of samples in the second dimension
    """

    def __init__(self, x1_size: int = 50, x2_size: int = 50):
        self.x1_size = x1_size
        self.x2_size = x2_size

        # define the range of x1 and x2 values
        x1_range = np.linspace(0, 2 * np.pi, x1_size)
        x2_range = np.linspace(0, 2 * np.pi, x2_size)

        # Create a grid of x1 and x2 values
        x1, x2 = np.meshgrid(x1_range, x2_range)

        # Calculate the function values
        y = np.sin(1.5 * x1) * np.sin(1.5 * x2)  # no noise

        # construct x as two dimensional vector observations
        x1 = x1.flatten()
        x2 = x2.flatten()
        self.x = np.c_[x1, x2]
        self.y = y.flatten()

        # Recast the data as torch tensors
        self.x = torch.FloatTensor(self.x)
        self.y = torch.FloatTensor(self.y)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx]


class Synthetic2DDataManager(DataManager):
    def __init__(
        self,
        dataset: torch.utils.data.Dataset[Tuple[torch.Tensor, torch.Tensor]],
        random_state: int = 42,
        numpy_flag: bool = False,
    ):
        self.numpy_flag = numpy_flag

        # Split the dataset into train and test
        indices = np.arange(2500)
        np.random.seed(random_state)
        np.random.shuffle(indices)
        train_indices = indices[: 2500 // 2]
        test_indices = indices[2500 // 2 :]

        # Split the train set into labelled and unlabelled
        num_labelled = len(train_indices) // 10  # about a tenth of the queryable set should be labelled
        labelled_indices = train_indices[:num_labelled]

        super().__init__(
            dataset=dataset,
            train_indices=train_indices.tolist(),
            test_indices=test_indices.tolist(),
            labelled_indices=labelled_indices.tolist(),
            loader_batch_size="full",
            loader_shuffle=False,
            loader_collate_fn=numpy_collate if numpy_flag else None,
        )
