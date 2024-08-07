import pyreadr
import torch

from pyrelational.datasets.base import BaseDataset
from pyrelational.datasets.download_utils import download_file

from .utils import create_splits, remap_to_int


class CreditCardDataset(BaseDataset):
    """Credit card fraud dataset, highly unbalanced and challenging.

    From Andrea Dal Pozzolo, Olivier Caelen, Reid A. Johnson, and Gianluca Bontempi.
    Calibrating probability with undersampling for unbalanced classification. In 2015
    IEEE Symposium Series on Computational Intelligence, pages 159–166, 2015.

    We use the original data from http://www.ulb.ac.be/di/map/adalpozz/data/creditcard.Rdata
    processed using pyreadr

    :param data_dir: path where to save the raw data default to /tmp/
    :param n_splits: an int describing the number of class stratified
            splits to compute
    :param random_seed: random seed for reproducibility on splits
    """

    raw_url = "http://www.ulb.ac.be/di/map/adalpozz/data/creditcard.Rdata"

    def __init__(self, data_dir: str = "/tmp/", n_splits: int = 5, random_seed: int = 0):
        super().__init__(n_splits=n_splits, random_seed=random_seed)
        self.data_dir = data_dir
        self.n_splits = n_splits
        self._load_dataset()

    def _load_dataset(self) -> None:
        download_file(self.raw_url, self.data_dir)
        file_name = self.raw_url.split("/")[-1]
        data = pyreadr.read_r(self.data_dir + file_name)

        data = data["creditcard"]
        data.reset_index(inplace=True)
        xcols = data.columns[1:-1]
        self.x = torch.from_numpy(data[xcols].to_numpy()).float()
        self.y = remap_to_int(torch.from_numpy(data["Class"].to_numpy().astype(int)))
        self.data_splits = create_splits(self.x, self.y, self.n_splits, self.random_seed)
