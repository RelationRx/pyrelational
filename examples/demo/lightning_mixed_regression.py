"""
TO DO: illustrate with dataset
"""

import torch

from pyrelational.informativeness import (
    regression_least_confidence,
    representative_sampling,
)
from pyrelational.strategies.strategy import Strategy


class MixedStrategy(Strategy):
    """
    Implements a strategy that combines least_confidence scorer with representative sampling.
    To this end, 10 times more samples than requested are selected based on least_confidence scorer,
    the list is then reduced based on representative_sampling.
    """

    def __init(self, datamanager, model):
        super(MixedStrategy, self).__init__(datamanager, model)

    def active_learning_step(self, num_annotate):
        self.model.train(self.l_loader, self.valid_loader)
        output = self.model(self.u_loader)
        scores = regression_least_confidence(x=output)
        ixs = torch.argsort(scores, descending=True).tolist()
        ixs = [self.u_indices[i] for i in ixs[: 10 * num_annotate]]
        subquery = torch.stack(self.data_manager.get_sample_feature_vectors(ixs))
        new_ixs = representative_sampling(subquery)
        return [ixs[i] for i in new_ixs]
