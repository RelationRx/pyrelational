"""
TO DO: illustrate with dataset
"""

import torch

from pyrelational.informativeness import StandardDeviation
from pyrelational.samplers.samplers import DeterministicSampler
from pyrelational.strategies.regression.regression_strategy import RegressionStrategy
from pyrelational.strategies.task_agnostic.representative_sampling_strategy import (
    representative_sampling,
)


class MixedStrategy(RegressionStrategy):
    """Implements a strategy that combines least_confidence scorer with representative sampling.
    To this end, 10 times more samples than requested are selected based on least_confidence scorer,
    the list is then reduced based on representative_sampling.
    """

    def __init__(self, clustering_method: str, oversample_factor: int = 10):
        super().__init__(StandardDeviation(), DeterministicSampler())
        self.clustering_method = clustering_method
        self.oversample_factor = oversample_factor

    def __call__(self, num_annotate, data_manager, model_manager):
        ixs = super().__call__(num_annotate * self.oversample_factor, data_manager, model_manager)
        subquery = torch.stack(data_manager.get_sample_feature_vectors(ixs))
        new_ixs = representative_sampling(subquery, num_annotate, self.clustering_method)
        return [ixs[i] for i in new_ixs]
