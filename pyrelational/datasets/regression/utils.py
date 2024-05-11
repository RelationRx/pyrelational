from typing import List

import numpy as np
from numpy.typing import NDArray
from sklearn.model_selection import KFold
from torch import Tensor


def create_splits(x: Tensor, y: Tensor, n_splits: int, random_seed: int) -> List[NDArray[np.int_]]:
    """
    Create stratified k-fold splits for the dataset using the dataset's features and labels.
    """
    skf = KFold(n_splits=n_splits, random_state=random_seed, shuffle=True)
    return list(skf.split(x.numpy(), y.numpy()))
