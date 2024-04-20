import torch
from sklearn.datasets import load_breast_cancer, load_digits

from pyrelational.datasets.base import BaseDataset

from .utils import create_splits


class BreastCancerDataset(BaseDataset):
    """
    UCI ML Breast Cancer Wisconsin (Diagnostic) dataset handler.

    This dataset features measurements from digitized images of breast mass and uses these features to classify
    the observations as benign or malignant.

    :param n_splits: Number of stratified splits for cross-validation.
    :param random_seed: Seed for random number generator for reproducibility.
    """

    def __init__(self, n_splits: int = 5, random_seed: int = 0):
        super().__init__(n_splits=n_splits, random_seed=random_seed)
        self._load_data()

    def _load_data(self) -> None:
        """
        Load and preprocess the Breast Cancer dataset. This method handles the conversion of the dataset into tensors
        suitable for model input and sets up splits.
        """
        data, labels = load_breast_cancer(return_X_y=True)
        self.x = torch.tensor(data, dtype=torch.float)
        self.y = torch.tensor(labels, dtype=torch.long)
        self.data_splits = create_splits(self.x, self.y, self.n_splits, self.random_seed)


class DigitDataset(BaseDataset):
    """UCI ML hand-written digits datasets

    From C. Kaynak (1995) Methods of Combining Multiple Classifiers and
    Their Applications to Handwritten Digit Recognition, MSc Thesis,
    Institute of Graduate Studies in Science and Engineering, Bogazici
    University.

    :param n_splits: an int describing the number of class stratified
            splits to compute
    :param random_seed: int setting the random seed for reproducibility
    """

    def __init__(self, n_splits: int = 5, random_seed: int = 0):
        super().__init__(n_splits=n_splits, random_seed=random_seed)
        self._load_data()

    def _load_data(self) -> None:
        """
        Load and preprocess the Digit dataset. This method handles the conversion of the dataset into tensors
        suitable for model input and sets up splits.
        """
        sk_x, sk_y = load_digits(return_X_y=True)
        self.x = torch.FloatTensor(sk_x)
        self.y = torch.LongTensor(sk_y)
        self.data_splits = create_splits(self.x, self.y, self.n_splits, self.random_seed)
