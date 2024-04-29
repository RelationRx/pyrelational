from typing import List

import numpy as np
from numpy.typing import NDArray
from sklearn.model_selection import StratifiedKFold
from torch import Tensor


def remap_to_int(torch_class_array: Tensor) -> Tensor:
    """
    Remap the elements in a torch tensor to contiguous integers starting from 0,
    which is useful for classification tasks where class labels should start from zero and be contiguous.

    :param torch_class_array: A torch.Tensor containing class labels, possibly non-integer or non-contiguous.
    :return: A torch.Tensor with class labels remapped to integers starting from 0.

    Example:
        >>> torch_class_array = torch.tensor([10, 10, 20, 20, 30])
        >>> remap_to_int(torch_class_array)
        tensor([0, 0, 1, 1, 2])
    """
    remapped_labels: Tensor = torch_class_array.unique(return_inverse=True)[1]
    return remapped_labels


def create_splits(x: Tensor, y: Tensor, n_splits: int, random_seed: int) -> List[NDArray[np.int_]]:
    """
    Create stratified k-fold splits for the dataset using the dataset's features and labels.
    """
    skf = StratifiedKFold(n_splits=n_splits, random_state=random_seed, shuffle=True)
    return list(skf.split(x.numpy(), y.numpy()))
