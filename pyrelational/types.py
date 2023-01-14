from typing import Any, Sized

from torch.utils.data import Dataset


class SizedDataset(Dataset[Any], Sized):
    ...
