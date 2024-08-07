"""
This module contains methods for scoring samples based on distances between
featurization of samples. These scorers are task-agnostic.
"""

import inspect
import logging
from typing import Any, Callable, List, Optional, Union, get_args

import numpy as np
import sklearn.cluster as sklust
import torch
from numpy.typing import NDArray
from sklearn.base import ClusterMixin
from sklearn.metrics import pairwise_distances_argmin, pairwise_distances_argmin_min
from torch import Tensor
from torch.utils.data import DataLoader

logging.basicConfig()
logger = logging.getLogger()

Array = Union[Tensor, NDArray[Any], List[Any]]


def relative_distance(
    query_set: Union[Array, DataLoader[Any]],
    reference_set: Union[Array, DataLoader[Any]],
    metric: Optional[Union[str, Callable[..., Any]]] = "euclidean",
    axis: int = 1,
) -> Tensor:
    """
    Function that return the minimum distance, according to input metric, from each sample in the query_set to the
    samples in the reference set.

    :param query_set: input containing the features of samples in the queryable pool. query set should either be an
        array-like object or a pytorch dataloader whose first element in each bactch is a featurisation of the samples
        in the batch.
    :param reference_set: input containing the features of samples already queried samples against which the distances
        are computed. reference set should either be an array-like object or a pytorch dataloader whose first element in
        each bactch is a featurisation of the samples in the batch.
    :param metric: defines the metric to be used to compute the distance. This should be supported by scikit-learn
        pairwise_distances function.
    :param axis: integer indicating which dimension the features are
    :return: pytorch tensor of dimension the number of samples in query_set containing the minimum distance from each
        sample to the reference set
    """
    if isinstance(query_set, (Tensor, np.ndarray, list)):
        query_set = np.array(query_set)
        query_set = query_set.reshape((query_set.shape[0], -1))

    if isinstance(reference_set, (Tensor, np.ndarray, list)):
        reference_set = np.array(reference_set)
        reference_set = reference_set.reshape((reference_set.shape[0], -1))

    if isinstance(reference_set, np.ndarray) and isinstance(query_set, np.ndarray):

        _, distances = pairwise_distances_argmin_min(query_set, reference_set, metric=metric, axis=axis)
    elif isinstance(reference_set, np.ndarray) and isinstance(query_set, DataLoader):
        distances = []
        for q in query_set:
            q = q[0].reshape((q[0].shape[0], -1))
            distances.append(pairwise_distances_argmin_min(q, reference_set, metric=metric, axis=axis)[1])
        distances = np.hstack(distances)

    elif isinstance(reference_set, DataLoader) and isinstance(query_set, np.ndarray):
        distances = []
        for r in reference_set:
            r = r[0].reshape((r[0].shape[0], -1))
            distances.append(pairwise_distances_argmin_min(query_set, r, metric=metric, axis=axis)[1])
        distances = np.min(np.vstack(distances), axis=0)

    elif isinstance(reference_set, DataLoader) and isinstance(query_set, DataLoader):
        distances = []
        for q in query_set:
            temp = []
            q = q[0].reshape((q[0].shape[0], -1))
            for r in reference_set:
                r = r[0].reshape((r[0].shape[0], -1))
                temp.append(pairwise_distances_argmin_min(q, r, metric=metric, axis=axis)[1])
            distances.append(np.min(np.vstack(temp), axis=0))
        distances = np.hstack(distances)

    else:
        raise TypeError("reference_set and query_set should either be an array_like structure or a pytorch DataLoader")

    return torch.from_numpy(distances).float()


def representative_sampling(
    query_set: Union[Array, DataLoader[Any]],
    num_annotate: int,
    clustering_method: Union[str, ClusterMixin] = "KMeans",
    **clustering_kwargs: Optional[Any],
) -> List[int]:
    """
    Function that selects representative samples of the query set. Representative selection relies on clustering
    algorithms in scikit-learn.

    :param query_set: input containing the features of samples in the queryable pool. query set should either be an
        array-like object or a pytorch dataloader whose first element in each bactch is a featurisation of the samples
        in the batch
    :param num_annotate: number of representative samples to identify
    :param clustering_method: name, or instantiated class, of the clustering method to use
    :param clustering_kwargs: arguments to be passed to instantiate clustering class if a string is passed to
        clustering_method
    :return: array-like containing the indices of the representative samples identified
    """

    if isinstance(query_set, DataLoader):
        out = []
        for q in query_set:
            out.append(q[0].reshape((q[0].shape[0], -1)))
        query_set = torch.cat(out, 0)
    query_set = np.array(query_set)

    if num_annotate >= query_set.shape[0]:  # if there are less samples than sought queries, return everything
        ret: List[int] = np.arange(query_set.shape[0]).tolist()
        return ret

    if isinstance(clustering_method, str) and hasattr(sklust, clustering_method):
        clustering_method = getattr(sklust, clustering_method)
        if "n_clusters" in inspect.getfullargspec(clustering_method).args:
            clustering_kwargs["n_clusters"] = num_annotate
        clustering_cls = clustering_method(**clustering_kwargs)
    elif isinstance(clustering_method, str):
        raise ValueError(f"{clustering_method} is not part of the sklearn package")
    elif isinstance(clustering_method, ClusterMixin):
        clustering_cls = clustering_method
    else:
        raise TypeError(
            """clustering_method argument type not supported, it should be either a string pointing to a method of
            sklearn or an instantiated clustering algorithm subclassing sklearn ClusterMixin"""
        )

    lbls = clustering_cls.fit_predict(query_set)
    if hasattr(clustering_cls, "cluster_centers_indices_"):
        indices: List[int] = clustering_cls.cluster_centers_indices_
        representative_samples = indices
    elif hasattr(clustering_cls, "cluster_centers_"):
        representative_samples = get_closest_query_to_centroids(clustering_cls.cluster_centers_, query_set, lbls)
    else:
        logger.warning(
            """Clustering method does not return centroids to identify closest samples,
            returning random sample from each cluster"""
        )
        representative_samples = get_random_query_from_cluster(lbls)

    num_samples = min(num_annotate, len(representative_samples))
    ret = np.random.choice(  # in case there are more that num_annotates samples
        representative_samples,
        size=(num_samples,),
        replace=False,
    ).tolist()
    return ret


def get_closest_query_to_centroids(
    centroids: NDArray[np.float_],
    query: NDArray[np.float_],
    cluster_assignment: NDArray[np.int_],
) -> List[int]:
    """
    Find the closest sample in query to centroids.

    :param centroids: array containing centroids
    :param query: array containing query samples
    :param cluster_assignment: indicate what cluster each query sample is associated with
    :return: list of indices of query samples
    """
    out = []
    for i in np.unique(cluster_assignment):
        ixs = np.where(cluster_assignment == i)[0]
        centroid = centroids[i].reshape(1, -1)
        subquery = query[ixs]
        j = pairwise_distances_argmin(centroid, subquery).item()
        out.append(ixs[j])
    return out


def get_random_query_from_cluster(cluster_assignment: NDArray[np.int_]) -> List[int]:
    """
    Get random indices drawn from each cluster.

    :param cluster_assignment: array indicating what cluster each sample is associated with.
    :return: list of indices of query samples
    """
    out = []
    for i in np.unique(cluster_assignment):
        ixs = np.where(cluster_assignment == i)[0]
        out.append(np.random.choice(ixs))
    return out
