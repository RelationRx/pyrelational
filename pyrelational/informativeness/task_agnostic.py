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
from sklearn.base import ClusterMixin
from sklearn.metrics import pairwise_distances_argmin, pairwise_distances_argmin_min
from torch.utils.data import DataLoader

logging.basicConfig()
logger = logging.getLogger()

Array = Union[torch.Tensor, np.ndarray, List]


def relative_distance(
    query_set: Union[Array, DataLoader],
    reference_set: Union[Array, DataLoader],
    metric: Optional[Union[str, Callable]] = "euclidean",
    axis: int = -1,
) -> torch.Tensor:
    """
    Function that return the minimum distance, according to input metric, from each sample in the query_set to the
    samples in the reference set.

    :param query_set: input containing the features of samples in the queryable pool. query set should either be an
    array-like object or a pytorch dataloader whose first element in each bactch is a featurisation of the samples in
    the batch.
    :param reference_set: input containing the features of samples already queried samples against which the distances
    are computed. reference set should either be an array-like object or a pytorch dataloader whose first element in
    each bactch is a featurisation of the samples in the batch.
    :param metric: defines the metric to be used to compute the distance. This should be supported by scikit-learn
    pairwise_distances function.
    :param axis: integer indicating which dimension the features are
    :return: pytorch tensor of dimension the number of samples in query_set containing the minimum distance from each
    sample to the reference set
    """
    if isinstance(query_set, get_args(Array)):
        query_set = np.array(query_set)
        query_set = query_set.reshape((query_set.shape[0], -1))

    if isinstance(reference_set, get_args(Array)):
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
    query_set: Union[Array, DataLoader],
    num_annotate: Optional[int] = None,
    clustering_method: Union[str, ClusterMixin] = "KMeans",
    **clustering_kwargs: Optional[Any],
) -> Array:
    """
    Function that selects representative samples of the query set. Representative selection relies on clustering
    algorithms in scikit-learn.

    :param query_set: input containing the features of samples in the queryable pool. query set should either be an
    array-like object or a pytorch dataloader whose first element in each bactch is a featurisation of the samples in
    the batch.
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

    if num_annotate is None and hasattr(clustering_method, "n_clusters"):
        num_annotate = clustering_method.n_clusters

    if num_annotate is not None and (
        num_annotate >= query_set.shape[0]
    ):  # if there are less samples than sought queries, return everything
        return np.arange(query_set.shape[0])

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
        return clustering_cls.cluster_centers_indices_
    elif hasattr(clustering_cls, "cluster_centers_"):
        return get_closest_query_to_centroids(clustering_cls.cluster_centers_, query_set, lbls)
    else:
        logger.warning(
            """Clustering method does not return centroids to identify closest samples,
            returning random sample from each cluster"""
        )
        return get_random_query_from_cluster(lbls)


def get_closest_query_to_centroids(centroids: np.ndarray, query: np.ndarray, cluster_assignment: np.ndarray) -> List:
    out = []
    for i in np.unique(cluster_assignment):
        ixs = np.where(cluster_assignment == i)[0]
        centroid = centroids[i].reshape(1, -1)
        subquery = query[ixs]
        j = pairwise_distances_argmin(centroid, subquery).item()
        out.append(ixs[j])
    return out


def get_random_query_from_cluster(cluster_assignment: np.ndarray) -> List:
    out = []
    for i in np.unique(cluster_assignment):
        ixs = np.where(cluster_assignment == i)[0]
        out.append(np.random.choice(ixs))
    return out
