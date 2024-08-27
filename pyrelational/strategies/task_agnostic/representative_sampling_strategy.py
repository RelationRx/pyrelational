"""Representative sampling based active learning strategy."""

import inspect
import warnings
from typing import Any, List, Union

import numpy as np
import sklearn.cluster as sklust
import torch
from numpy.typing import NDArray
from sklearn.base import ClusterMixin
from sklearn.metrics import pairwise_distances_argmin
from torch.utils.data import DataLoader

from pyrelational.data_managers import DataManager
from pyrelational.informativeness.task_agnostic_scorers import ARRAY
from pyrelational.strategies.abstract_strategy import Strategy


class RepresentativeSamplingStrategy(Strategy):
    """Representative sampling based active learning strategy."""

    def __init__(
        self,
        clustering_method: Union[str, ClusterMixin] = "KMeans",
        **clustering_kwargs: Any,
    ):
        """Initialise the strategy with a clustering method and its arguments.

        :param clustering_method: name, or instantiated class, of the clustering method to use
        :param clustering_kwargs: arguments to be passed to instantiate clustering class if a string is passed to
            clustering_method
        """
        self.clustering_method = clustering_method
        self.clustering_kwargs = clustering_kwargs

    def __call__(
        self,
        data_manager: DataManager,
        num_annotate: int,
    ) -> List[int]:
        """Identify samples for labelling based on representative sampling informativeness measure.

        :param data_manager: A pyrelational data manager
            which keeps track of what has been labelled and creates data loaders for
            active learning
        :param num_annotate: number of samples to annotate

        :return: list of indices to annotate
        """
        unlabelled_features = torch.stack(data_manager.get_sample_feature_vectors(data_manager.u_indices))
        representative_samples = representative_sampling(
            unlabelled_features,
            num_annotate=num_annotate,
            clustering_method=self.clustering_method,
            **self.clustering_kwargs,
        )
        return [data_manager.u_indices[i] for i in representative_samples]


def representative_sampling(
    query_set: Union[ARRAY, DataLoader[Any]],
    num_annotate: int,
    clustering_method: Union[str, ClusterMixin] = "KMeans",
    **clustering_kwargs: Any,
) -> List[int]:
    """
    Select representative samples from the query set using clustering algorithms from scikit-learn.

    :param query_set: The query set, either as an array-like object or a PyTorch DataLoader.
                      If a DataLoader, the first element of each batch should be the features of the samples.
    :param num_annotate: Number of representative samples to select.
    :param clustering_method: The clustering method to use, either as a string (name of the clustering algorithm)
                              or as an instantiated clustering class.
    :param clustering_kwargs: Additional arguments for the clustering method, used if clustering_method is a string.
    :return: A list of indices representing the selected samples.
    """
    query_set_array = _convert_query_set_to_array(query_set)

    if num_annotate >= len(query_set_array):
        return list(range(len(query_set_array)))

    clustering_cls = _initialize_clustering_method(clustering_method, num_annotate, **clustering_kwargs)

    labels = clustering_cls.fit_predict(query_set_array)
    representative_samples = _get_representative_samples(clustering_cls, query_set_array, labels)

    return _select_final_samples(representative_samples, num_annotate)


def _convert_query_set_to_array(query_set: Union[ARRAY, DataLoader[Any]]) -> NDArray[np.float_]:
    """
    Convert the query set to a NumPy array.

    :param query_set: The query set, either as a NumPy array or a PyTorch DataLoader.
    :return: The query set as a NumPy array.
    """
    if isinstance(query_set, DataLoader):
        features = [batch[0].reshape(batch[0].shape[0], -1) for batch in query_set]
        query_set = torch.cat(features, dim=0)
    return np.array(query_set)


def _initialize_clustering_method(
    clustering_method: Union[str, ClusterMixin], num_annotate: int, **clustering_kwargs: Any
) -> ClusterMixin:
    """
    Initialize the clustering method.

    :param clustering_method: The clustering method to use, either as a string or an instantiated class.
    :param num_annotate: Number of clusters (representative samples) to create.
    :param clustering_kwargs: Additional arguments for the clustering method.
    :return: An instantiated clustering class.
    """
    if isinstance(clustering_method, str):
        if hasattr(sklust, clustering_method):
            clustering_cls = getattr(sklust, clustering_method)
            if "n_clusters" in inspect.getfullargspec(clustering_cls).args:
                clustering_kwargs["n_clusters"] = num_annotate
            return clustering_cls(**clustering_kwargs)
        else:
            raise ValueError(f"{clustering_method} is not part of the sklearn package")
    elif isinstance(clustering_method, ClusterMixin):
        return clustering_method
    else:
        raise TypeError(
            "clustering_method should be either a string name of a sklearn clustering method "
            "or an instantiated ClusterMixin subclass."
        )


def _get_representative_samples(
    clustering_cls: ClusterMixin, query_set: NDArray[np.float_], labels: NDArray[np.int_]
) -> List[int]:
    """
    Get representative samples based on the clustering method.

    :param clustering_cls: The clustering method used.
    :param query_set: The query set.
    :param labels: Cluster labels for each sample in the query set.
    :return: A list of indices representing the representative samples.
    """
    if hasattr(clustering_cls, "cluster_centers_indices_"):
        ret: List[int] = clustering_cls.cluster_centers_indices_.tolist()
        return ret
    elif hasattr(clustering_cls, "cluster_centers_"):
        return _get_closest_query_to_centroids(clustering_cls.cluster_centers_, query_set, labels)
    else:
        warnings.warn(
            "Clustering method does not return centroids. Returning a random sample from each cluster.",
            stacklevel=3,
        )
        return _get_random_query_from_cluster(labels)


def _get_closest_query_to_centroids(
    centroids: NDArray[np.float_], query_set: NDArray[np.float_], labels: NDArray[np.int_]
) -> List[int]:
    """
    Find the closest sample in the query set to each centroid.

    :param centroids: Array containing the cluster centroids.
    :param query_set: Array containing the query samples.
    :param labels: Array indicating the cluster assignment for each query sample.
    :return: List of indices of the closest query samples to the centroids.
    """
    representative_samples = []
    for cluster_id in np.unique(labels):
        cluster_indices = np.where(labels == cluster_id)[0]
        subquery = query_set[cluster_indices]
        closest_idx = pairwise_distances_argmin(centroids[cluster_id].reshape(1, -1), subquery).item()
        representative_samples.append(cluster_indices[closest_idx])
    return representative_samples


def _get_random_query_from_cluster(labels: NDArray[np.int_]) -> List[int]:
    """
    Select random samples from each cluster.

    :param labels: Array indicating the cluster assignment for each query sample.
    :return: List of indices of randomly selected samples from each cluster.
    """
    return [np.random.choice(np.where(labels == cluster_id)[0]) for cluster_id in np.unique(labels)]


def _select_final_samples(representative_samples: List[int], num_annotate: int) -> List[int]:
    """
    Select the final set of representative samples.

    :param representative_samples: List of candidate representative samples.
    :param num_annotate: The number of samples to annotate.
    :return: A list of indices representing the selected samples.
    """
    num_samples = min(num_annotate, len(representative_samples))
    ret: List[int] = np.random.choice(representative_samples, size=num_samples, replace=False).tolist()
    return ret
