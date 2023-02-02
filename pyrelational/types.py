from abc import ABC
from typing import Sized, TypeVar

from torch.utils.data import Dataset

T = TypeVar("T")


class SizedDataset(Dataset[T], Sized, ABC):
    ...
