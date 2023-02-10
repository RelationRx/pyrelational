"""Representative sampling based active learning strategy
"""
from typing import Any, List, Union

import numpy as np
import torch
from sklearn.base import ClusterMixin

from pyrelational.data_managers import DataManager
from pyrelational.informativeness import representative_sampling
from pyrelational.strategies.abstract_strategy import Strategy


class RepresentativeSamplingStrategy(Strategy):
    """Representative sampling based active learning strategy"""

    def __init__(
        self,
        clustering_method: Union[str, ClusterMixin] = "KMeans",
        **clustering_kwargs: Any,
    ):
        """
        :param clustering_method: name, or instantiated class, of the clustering method to use
        :param clustering_kwargs: arguments to be passed to instantiate clustering class if a string is passed to
            clustering_method
        """
        super(RepresentativeSamplingStrategy, self).__init__()
        self.clustering_method = clustering_method
        self.clustering_kwargs = clustering_kwargs

    def __call__(
        self,
        data_manager: DataManager,
        num_annotate: int,
    ) -> List[int]:
        """
        Call function which identifies samples which need to be labelled

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
