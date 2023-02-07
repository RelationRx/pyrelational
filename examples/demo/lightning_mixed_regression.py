"""
TO DO: illustrate with dataset
"""

import torch

from pyrelational.informativeness import (
    regression_least_confidence,
    representative_sampling,
)
from pyrelational.strategies.abstract_strategy import Strategy


class MixedStrategy(Strategy):
    """
    Implements a strategy that combines least_confidence scorer with representative sampling.
    To this end, 10 times more samples than requested are selected based on least_confidence scorer,
    the list is then reduced based on representative_sampling.
    """

    def __call__(self, num_annotate, data_manager, model_manager):
        output = self.train_and_infer(data_manager=data_manager, model_manager=model_manager)
        scores = regression_least_confidence(x=output.squeeze(-1))
        ixs = torch.argsort(scores, descending=True).tolist()
        ixs = [data_manager.u_indices[i] for i in ixs[: 10 * num_annotate]]
        subquery = torch.stack(data_manager.get_sample_feature_vectors(ixs))
        new_ixs = representative_sampling(subquery)
        return [ixs[i] for i in new_ixs]
