from typing import Tuple

from torch import Tensor
from torch.utils.data import Dataset


class SimpleDataset(Dataset[Tuple[Tensor, Tensor]]):
    def __init__(self, x: Tensor, y: Tensor):
        """
        Instantiate simple dataset.

        :param x: feature tensor
        :param y: target tensor
        """
        self.x = x
        self.y = y

    def __getitem__(self, item: int) -> Tuple[Tensor, Tensor]:
        """Get dataset item."""
        return self.x[item], self.y[item]

    def __len__(self) -> int:
        """Return length of dataset."""
        return len(self.x)
