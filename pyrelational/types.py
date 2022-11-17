from typing import Sized

from torch.utils.data import Dataset


class SizedDataset(Dataset, Sized):
    ...
