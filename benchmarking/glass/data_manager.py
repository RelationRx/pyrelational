# type: ignore

"""Benchmarking DataManager for the glass dataset
"""
import random
import time
from typing import Any, Dict, List, Optional, Type, TypeVar, Union

import numpy as np
import torch
from numpy.typing import NDArray

from pyrelational.data_managers import DataManager
from pyrelational.datasets.classification.uci import UCIGlass

from ..classification_experiment_utils import (
    make_class_stratified_train_val_test_split,
    pick_one_sample_per_class,
)


def get_glass_data_manager() -> DataManager:
    # Add a random wait between 1 and 10 seconds to avoid race conditions
    # when creating the DataManager
    time.sleep(random.randint(1, 10))
    ds = UCIGlass()
    train_indices, valid_indices, test_indices = make_class_stratified_train_val_test_split(ds, k=5)
    return DataManager(
        ds,
        train_indices=train_indices,
        validation_indices=valid_indices,
        test_indices=test_indices,
        labelled_indices=pick_one_sample_per_class(ds, train_indices),
        loader_batch_size="full",
        loader_collate_fn=numpy_collate,
    )


def numpy_collate(
    batch: List[Union[torch.Tensor, NDArray[Union[Any, np.float32, np.float64]]]]
) -> List[NDArray[Union[Any, np.float32, np.float64]]]:
    """Collate function for a Pytorch to Numpy DataLoader"""
    return [np.stack([b.numpy() if isinstance(b, torch.Tensor) else b for b in samples]) for samples in zip(*batch)]
