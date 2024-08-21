"""This module contains methods for scoring samples based on distances between featurization of samples.

These scorers are task-agnostic.
"""

# import inspect
import logging

# import struct
from typing import Any, List, Union  # , get_args, Callable,  Optional,

import numpy as np

# import sklearn.cluster as sklust
import torch
from numpy.typing import NDArray

# from sklearn.base import ClusterMixin
from sklearn.metrics import pairwise_distances_argmin_min  # pairwise_distances_argmin,
from torch import Tensor
from torch.utils.data import DataLoader

from pyrelational.informativeness.abstract_scorers import AbstractScorer

logging.basicConfig()
logger = logging.getLogger()

Array = Union[Tensor, NDArray[Any], List[Any]]


class RelativeDistanceScorer(AbstractScorer):
    """Relative distance scorer.

    It computes the minimum distance from each sample in the query set to the samples in the reference set.
    """

    def __init__(self, metric: str = "euclidean", axis: int = 1) -> None:
        """
        Initialize the scorer with the metric and axis.

        :param metric: defines the metric to be used to compute the distance. This should be supported by scikit-learn
            pairwise_distances function.
        :param axis: integer indicating which dimension the features are
        """
        self.metric = metric
        self.axis = axis

    def __call__(
        self, query_set: Union[Array, DataLoader[Any]], reference_set: Union[Array, DataLoader[Any]]
    ) -> Tensor:
        """
        Compute the relative minimum distances between the query and reference sets.

        :param query_set: input containing the features of samples in the queryable pool.
        :param reference_set: input containing the features of samples already queried.
        :return: pytorch tensor of distances from each sample in the query set to the
            closest sample in the reference set.
        """
        query_set = self._prepare_set(query_set)
        reference_set = self._prepare_set(reference_set)

        if isinstance(query_set, np.ndarray) and isinstance(reference_set, np.ndarray):
            distances = self._compute_distances(query_set, reference_set)

        elif isinstance(query_set, DataLoader) and isinstance(reference_set, np.ndarray):
            distances = self._compute_distances_dataloader_to_array(query_set, reference_set)

        elif isinstance(query_set, np.ndarray) and isinstance(reference_set, DataLoader):
            distances = self._compute_distances_array_to_dataloader(query_set, reference_set)

        elif isinstance(query_set, DataLoader) and isinstance(reference_set, DataLoader):
            distances = self._compute_distances_dataloader_to_dataloader(query_set, reference_set)

        else:
            raise TypeError(
                "reference_set and query_set should either be an array-like structure or a PyTorch DataLoader"
            )

        return torch.from_numpy(distances).float()

    @staticmethod
    def _prepare_set(data_set: Union[Array, DataLoader[Any]]) -> Union[NDArray[Any], DataLoader[Any]]:
        """Convert the input set to a 2D numpy array."""
        if isinstance(data_set, (Tensor, np.ndarray, list)):
            data_set = np.array(data_set)
            return data_set.reshape((data_set.shape[0], -1))
        elif isinstance(data_set, DataLoader):
            return data_set
        else:
            raise TypeError("Input set should either be an array-like structure or a PyTorch DataLoader")

    def _compute_distances(self, query: NDArray[Any], reference: NDArray[Any]) -> NDArray[Any]:
        """Compute the minimum distances from query samples to reference samples."""
        distances: NDArray[Any] = pairwise_distances_argmin_min(query, reference, metric=self.metric, axis=self.axis)[1]
        return distances

    def _compute_distances_dataloader_to_array(
        self, query_loader: DataLoader[Any], reference_array: NDArray[Any]
    ) -> NDArray[Any]:
        """Compute distances when query set is a DataLoader and reference set is a numpy array."""
        distances: List[NDArray[Any]] = []
        for batch in query_loader:
            query = batch[0].reshape((batch[0].shape[0], -1))
            distances.append(self._compute_distances(query, reference_array))
        return np.hstack(distances)

    def _compute_distances_array_to_dataloader(
        self, query_array: NDArray[Any], reference_loader: DataLoader[Any]
    ) -> NDArray[Any]:
        """Compute distances when query set is a numpy array and reference set is a DataLoader."""
        distances: List[NDArray[Any]] = []
        for batch in reference_loader:
            reference = batch[0].reshape((batch[0].shape[0], -1))
            distances.append(self._compute_distances(query_array, reference))
        ret: NDArray[Any] = np.min(np.vstack(distances), axis=0)
        return ret

    def _compute_distances_dataloader_to_dataloader(
        self, query_loader: DataLoader[Any], reference_loader: DataLoader[Any]
    ) -> NDArray[Any]:
        """Compute distances when both query and reference sets are DataLoaders."""
        distances: List[NDArray[Any]] = []
        for query_batch in query_loader:
            temp_distances: List[NDArray[Any]] = []
            query = query_batch[0].reshape((query_batch[0].shape[0], -1))
            for reference_batch in reference_loader:
                reference = reference_batch[0].reshape((reference_batch[0].shape[0], -1))
                temp_distances.append(self._compute_distances(query, reference))
            distances.append(np.min(np.vstack(temp_distances), axis=0))
        return np.hstack(distances)
