from pyrelational.datasets.base import BaseDataset
from pyrelational.datasets.uci_datasets import UCIDatasets


class UCIRegression(BaseDataset):
    """UCI regression dataset base class

    :param n_splits: an int describing the number of class stratified
        splits to compute
    """

    def __init__(self, name: str, data_dir: str = "/tmp/", n_splits: int = 5, random_seed: int = 0):
        super().__init__(n_splits=n_splits, random_seed=random_seed)
        dataset = UCIDatasets(name=name, data_dir=data_dir, n_splits=n_splits)
        self.data_dir = dataset.data_dir
        self.name = dataset.name
        self.data_splits = dataset.data_splits

        x, y = dataset.get_data()
        self.len_dataset = len(x)
        self.x = x
        self.y = y


class UCIConcrete(UCIRegression):
    """UCI housing dataset

    :param n_splits: an int describing the number of class stratified
        splits to compute
    """

    def __init__(self, data_dir: str = "/tmp/", n_splits: int = 5):
        super(UCIConcrete, self).__init__(name="concrete", data_dir=data_dir, n_splits=n_splits)


class UCIEnergy(UCIRegression):
    """UCI housing dataset

    :param n_splits: an int describing the number of class stratified
        splits to compute
    """

    def __init__(self, data_dir: str = "/tmp/", n_splits: int = 5):
        super(UCIEnergy, self).__init__(name="energy", data_dir=data_dir, n_splits=n_splits)


class UCIPower(UCIRegression):
    """UCI housing dataset

    :param n_splits: an int describing the number of class stratified
        splits to compute
    """

    def __init__(self, data_dir: str = "/tmp/", n_splits: int = 5):
        super(UCIPower, self).__init__(name="power", data_dir=data_dir, n_splits=n_splits)


class UCIWine(UCIRegression):
    """UCI housing dataset

    :param n_splits: an int describing the number of class stratified
        splits to compute
    """

    def __init__(self, data_dir: str = "/tmp/", n_splits: int = 5):
        super(UCIWine, self).__init__(name="wine", data_dir=data_dir, n_splits=n_splits)


class UCIYacht(UCIRegression):
    """UCI housing dataset

    :param n_splits: an int describing the number of class stratified
        splits to compute
    """

    def __init__(self, data_dir: str = "/tmp/", n_splits: int = 5):
        super(UCIYacht, self).__init__(name="yacht", data_dir=data_dir, n_splits=n_splits)


class UCIAirfoil(UCIRegression):
    """UCI Airfoil dataset

    :param n_splits: an int describing the number of class stratified
        splits to compute
    """

    def __init__(self, data_dir: str = "/tmp/", n_splits: int = 5):
        super(UCIAirfoil, self).__init__(name="airfoil", data_dir=data_dir, n_splits=n_splits)
