from typing import Tuple

from torch import Tensor
from torch.utils.data import Dataset


class BaseDataset(Dataset[Tuple[Tensor, Tensor]]):
    """
    A base class for all datasets to inherit from. It handles common functionalities such as
    initialization, downloading datasets, and creating stratified k-fold splits.

    :param n_splits: Number of splits for cross-validation.
    :param random_seed: Seed for random number generator for reproducibility.
    """

    x: Tensor
    y: Tensor

    def __init__(self, n_splits: int, random_seed: int):
        """
        Initialize the BaseDataset with the number of splits and a seed for reproducibility.

        :param n_splits: Number of splits for stratified k-fold.
        :param random_seed: Random seed for reproducibility.
        """
        super(BaseDataset, self).__init__()
        self.n_splits = n_splits
        self.random_seed = random_seed

    def __len__(self) -> int:
        """
        Return the total number of samples in the dataset.

        :return: Total number of samples.
        """
        return len(self.x)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        """
        Fetch the sample and its corresponding label at the given index.

        :param idx: Index of the sample to retrieve.
        :return: Tuple containing the sample and its label.
        """
        return self.x[idx], self.y[idx]
