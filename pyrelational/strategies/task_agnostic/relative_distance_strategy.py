from typing import List

import torch

from pyrelational.data_managers import DataManager
from pyrelational.informativeness import relative_distance
from pyrelational.strategies.abstract_strategy import Strategy


class RelativeDistanceStrategy(Strategy):
    """Diversity sampling based active learning strategy."""

    def __init__(self, metric: str = "euclidean"):
        """
        :param metric: Name of distance metric to use. This should be supported by scikit-learn
            pairwise_distances function.
        """
        super(RelativeDistanceStrategy, self).__init__()
        self.metric = metric

    def __call__(self, num_annotate: int, data_manager: DataManager) -> List[int]:
        """
        Call function which identifies samples which need to be labelled

        :param num_annotate: number of samples to annotate
        :param data_manager: A pyrelational data manager
            which keeps track of what has been labelled and creates data loaders for
            active learning

        :return: list of indices to annotate
        """
        scores = relative_distance(
            data_manager.get_unlabelled_loader(), data_manager.get_labelled_loader(), metric=self.metric
        )
        ixs = torch.argsort(scores, descending=True).tolist()
        return [data_manager.u_indices[i] for i in ixs[:num_annotate]]
