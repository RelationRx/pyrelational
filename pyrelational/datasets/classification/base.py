import os
import urllib.request
from typing import Tuple

from sklearn.model_selection import StratifiedKFold
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

    def _download_dataset(self, url: str, save_path: str) -> None:
        """
        Download the dataset from a URL if it doesn't exist at the specified path.

        :param url: URL from where to download the dataset.
        :param save_path: Full path to save the dataset file.
        """
        if not os.path.exists(save_path):
            os.mkdir(os.path.dirname(save_path))
            urllib.request.urlretrieve(url, save_path)

    def _create_splits(self) -> None:
        """
        Create stratified k-fold splits for the dataset using the dataset's features and labels.
        """
        skf = StratifiedKFold(n_splits=self.n_splits, random_state=self.random_seed, shuffle=True)
        self.data_splits = list(skf.split(self.x.numpy(), self.y.numpy()))
