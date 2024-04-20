import torch

from pyrelational.datasets.uci_datasets import UCIDatasets

from .base import BaseDataset
from .utils import remap_to_int


class UCIClassification(BaseDataset):
    """
    A generic class for handling UCI datasets, providing mechanisms to download, preprocess, and split the dataset.

    :param name: Identifier for the UCI dataset.
    :param data_dir: Directory where datasets are stored or will be downloaded.
    :param n_splits: Number of stratified splits for cross-validation.
    :param random_seed: Random seed for reproducibility of splits.
    """

    def __init__(self, name: str, data_dir: str = "/tmp/", n_splits: int = 5, random_seed: int = 0):
        super().__init__(n_splits=n_splits, random_seed=random_seed)
        self.name = name
        self.data_dir = data_dir
        self.dataset = UCIDatasets(name=self.name, data_dir=self.data_dir)
        self._load_data()

    def _load_data(self) -> None:
        """
        Load and preprocess the dataset. This involves loading the data using UCIDatasets,
        possibly transforming it, and preparing it for model training.
        """
        data, labels = self.dataset.get_data()
        self.x = torch.tensor(data, dtype=torch.float)
        self.y = torch.tensor(labels, dtype=torch.long)
        self.y = remap_to_int(self.y)  # Ensure labels are continuous and start from 0
        self._create_splits()


class UCIGlass(UCIClassification):
    """
    UCI Glass dataset for classification tasks.

    Inherits from UCIClassification and uses its mechanisms to load and preprocess the Glass dataset specifically.
    """

    def __init__(self, data_dir: str = "/tmp/", n_splits: int = 5, random_seed: int = 0):
        super().__init__(name="glass", data_dir=data_dir, n_splits=n_splits, random_seed=random_seed)


class UCIParkinsons(UCIClassification):
    """
    UCI Parkinsons dataset for classification tasks.

    Inherits from UCIClassification and uses its mechanisms to load and
    preprocess the Parkinsons dataset specifically.
    """

    def __init__(self, data_dir: str = "/tmp/", n_splits: int = 5, random_seed: int = 0):
        super().__init__(name="parkinsons", data_dir=data_dir, n_splits=n_splits, random_seed=random_seed)


class UCISeeds(UCIClassification):
    """
    UCI Seeds dataset for classification tasks.

    Inherits from UCIClassification and uses its mechanisms to load and
    preprocess the Seeds dataset specifically.
    """

    def __init__(self, data_dir: str = "/tmp/", n_splits: int = 5, random_seed: int = 0):
        super().__init__(name="seeds", data_dir=data_dir, n_splits=n_splits, random_seed=random_seed)
