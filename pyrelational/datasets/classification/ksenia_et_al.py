import numpy as np
import scipy
import torch

from pyrelational.datasets.download_utils import download_file

from .base import BaseDataset
from .utils import remap_to_int


class StriatumDataset(BaseDataset):
    """Striatum dataset as used in Konyushkova et al. 2017

    From Ksenia Konyushkova, Raphael Sznitman, Pascal Fua 'Learning Active
    Learning from Data', NIPS 2017

    :param data_dir: path where to save the raw data default to /tmp/
    :param n_splits: an int describing the number of class stratified
            splits to compute
    :param random_seed: random seed for reproducibility on splits
    """

    train_feat_url = "https://github.com/ksenia-konyushkova/LAL/raw/master/data/striatum_train_features_mini.mat"
    train_label_url = "https://github.com/ksenia-konyushkova/LAL/raw/master/data/striatum_train_labels_mini.mat"
    test_feat_url = "https://github.com/ksenia-konyushkova/LAL/raw/master/data/striatum_test_features_mini.mat"
    test_label_url = "https://github.com/ksenia-konyushkova/LAL/raw/master/data/striatum_test_labels_mini.mat"

    def __init__(self, data_dir: str = "/tmp/", n_splits: int = 5, random_seed: int = 0):
        super(StriatumDataset, self).__init__(n_splits=n_splits, random_seed=random_seed)
        self.data_dir = data_dir
        self.n_splits = n_splits
        self._load_dataset()

    def _load_dataset(self) -> None:
        """Download, process, and get stratified splits"""
        download_file(self.train_feat_url, self.data_dir)
        download_file(self.test_feat_url, self.data_dir)
        download_file(self.train_label_url, self.data_dir)
        download_file(self.test_label_url, self.data_dir)

        # process
        train_feat = (scipy.io.loadmat(self.data_dir + "striatum_train_features_mini.mat"))["features"]
        test_feat = scipy.io.loadmat(self.data_dir + "striatum_test_features_mini.mat")["features"]
        train_label = scipy.io.loadmat(self.data_dir + "striatum_train_labels_mini.mat")["labels"]
        test_label = scipy.io.loadmat(self.data_dir + "striatum_test_labels_mini.mat")["labels"]

        x = np.vstack([train_feat, test_feat])
        y = np.vstack([train_label, test_label])

        self.x = torch.from_numpy(x).float()
        self.y = torch.from_numpy(y).long().squeeze()
        self.y = remap_to_int(self.y).long()
        self._create_splits()


class GaussianCloudsDataset(BaseDataset):
    """GaussianClouds from Konyushkova et al. 2017 basically a imbalanced
    binary classification task created from multivariate gaussian blobs

    From Ksenia Konyushkova, Raphael Sznitman, Pascal Fua 'Learning Active
    Learning from Data', NIPS 2017

    :param data_dir: path where to save the raw data default to /tmp/
    :param n_splits: an int describing the number of class stratified
            splits to compute
    :param random_seed: random seed for reproducibility on splits
    """

    def __init__(
        self,
        data_dir: str = "/tmp/",
        n_splits: int = 5,
        random_seed: int = 0,
        size: int = 1000,
        n_dim: int = 2,
        random_balance: bool = False,
    ):
        super().__init__(n_splits=n_splits, random_seed=random_seed)
        self.data_dir = data_dir
        self.n_splits = n_splits
        self._create_dataset(
            size=size,
            n_dim=n_dim,
            random_balance=random_balance,
            random_seed=random_seed,
        )

    def _create_dataset(
        self, size: int = 1000, n_dim: int = 2, random_balance: bool = False, random_seed: int = 0
    ) -> None:
        rng = np.random.default_rng(random_seed)
        if random_balance:
            # proportion of class 1 to vary from 10% to 90%
            cl1_prop = rng.random()
            cl1_prop = (cl1_prop - 0.5) * 0.8 + 0.5
        else:
            cl1_prop = 0.8

        trainSize1 = int(size * cl1_prop)
        trainSize2 = size - trainSize1
        testSize1 = trainSize1 * 10
        testSize2 = trainSize2 * 10

        # Generate parameters of datasets
        mean1 = rng.random(n_dim)
        cov1 = rng.random((n_dim, n_dim)) - 0.5
        cov1 = np.dot(cov1, cov1.transpose())
        mean2 = rng.random(n_dim)
        cov2 = rng.random((n_dim, n_dim)) - 0.5
        cov2 = np.dot(cov2, cov2.transpose())

        # Training data generation
        trainX1 = rng.multivariate_normal(mean1, cov1, trainSize1)
        trainY1 = np.ones((trainSize1, 1))
        trainX2 = rng.multivariate_normal(mean2, cov2, trainSize2)
        trainY2 = np.zeros((trainSize2, 1))

        # Testing data generation
        testX1 = rng.multivariate_normal(mean1, cov1, testSize1)
        testY1 = np.ones((testSize1, 1))
        testX2 = rng.multivariate_normal(mean2, cov2, testSize2)
        testY2 = np.zeros((testSize2, 1))

        train_data = np.concatenate((trainX1, trainX2), axis=0)
        train_labels = np.concatenate((trainY1, trainY2))
        test_data = np.concatenate((testX1, testX2), axis=0)
        test_labels = np.concatenate((testY1, testY2))

        x = np.vstack([train_data, test_data])
        y = np.vstack([train_labels, test_labels]).squeeze()

        self.x = torch.from_numpy(x).float()
        self.y = torch.from_numpy(y).long().squeeze()
        self._create_splits()


class Checkerboard2x2Dataset(BaseDataset):
    """Checkerboard2x2 dataset from Konyushkova et al. 2017

    From Ksenia Konyushkova, Raphael Sznitman, Pascal Fua 'Learning Active
    Learning from Data', NIPS 2017

    :param data_dir: path where to save the raw data default to /tmp/
    :param n_splits: an int describing the number of class stratified
            splits to compute
    :param random_seed: random seed for reproducibility on splits
    """

    raw_train_url = "https://github.com/ksenia-konyushkova/LAL/raw/master/data/checkerboard2x2_train.npz"
    raw_test_url = "https://github.com/ksenia-konyushkova/LAL/raw/master/data/checkerboard2x2_test.npz"

    def __init__(self, data_dir: str = "/tmp/", n_splits: int = 5, random_seed: int = 0):
        super().__init__(n_splits=n_splits, random_seed=random_seed)
        self.data_dir = data_dir
        self.n_splits = n_splits
        self._load_dataset()

    def _load_dataset(self) -> None:
        """Download, process, and get stratified splits"""
        # download
        download_file(self.raw_train_url, self.data_dir)
        download_file(self.raw_test_url, self.data_dir)

        # process
        train = np.load(self.data_dir + "checkerboard2x2_train.npz")
        test = np.load(self.data_dir + "checkerboard2x2_test.npz")

        train_feat, train_label = train["x"], train["y"]
        test_feat, test_label = test["x"], test["y"]

        x = np.vstack([train_feat, test_feat])
        y = np.vstack([train_label, test_label])

        self.x = torch.from_numpy(x).float()
        self.y = torch.from_numpy(y).long().squeeze()
        self._create_splits()


class Checkerboard4x4Dataset(BaseDataset):
    """Checkerboard 4x4 dataset from Konyushkova et al. 2017

    From Ksenia Konyushkova, Raphael Sznitman, Pascal Fua 'Learning Active
    Learning from Data', NIPS 2017

    :param data_dir: path where to save the raw data default to /tmp/
    :param n_splits: an int describing the number of class stratified
            splits to compute
    :param random_seed: random seed for reproducibility on splits
    """

    train_url = "https://github.com/ksenia-konyushkova/LAL/raw/master/data/checkerboard4x4_train.npz"
    test_url = "https://github.com/ksenia-konyushkova/LAL/raw/master/data/checkerboard4x4_test.npz"

    def __init__(self, data_dir: str = "/tmp/", n_splits: int = 5, random_seed: int = 0):
        super().__init__(n_splits=n_splits, random_seed=random_seed)
        self.data_dir = data_dir
        self.n_splits = n_splits
        self._load_dataset()

    def _load_dataset(self) -> None:
        """Download, process, and get stratified splits"""
        # download
        download_file(self.train_url, self.data_dir)
        download_file(self.test_url, self.data_dir)

        # process
        train = np.load(self.data_dir + "checkerboard4x4_train.npz")
        test = np.load(self.data_dir + "checkerboard4x4_test.npz")

        train_feat, train_label = train["x"], train["y"]
        test_feat, test_label = test["x"], test["y"]

        x = np.vstack([train_feat, test_feat])
        y = np.vstack([train_label, test_label])

        self.x = torch.from_numpy(x).float()
        self.y = torch.from_numpy(y).long().squeeze()
        self._create_splits()
