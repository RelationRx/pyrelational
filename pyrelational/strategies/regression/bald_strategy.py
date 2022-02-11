from typing import List

import numpy as np
import torch

from pyrelational.data import GenericDataManager
from pyrelational.informativeness import regression_bald
from pyrelational.models import GenericModel
from pyrelational.strategies.regression.generic_regression_strategy import (
    GenericRegressionStrategy,
)


class BALDStrategy(GenericRegressionStrategy):
    """Implements BALD Strategy whereby unlabelled samples are queried based on mutual information score based on
    multiple estimator models."""

    def __init__(self, data_manager: GenericDataManager, model: GenericModel):
        super(BALDStrategy, self).__init__(data_manager, model)
        self.scoring_fn = regression_bald


class SoftBALDStrategy(BALDStrategy):
    """Implements soft BALD Strategy whereby unlabelled samples are queried based on mutual information score based on
    multiple estimator models. In contrast to Bald the query is drawn from unlabelled pool based on probabilities
    derived from bald scores instead of using an argmax operation"""

    def __init__(
        self,
        data_manager: GenericDataManager,
        model: GenericModel,
        temperature: float = 0.5,
    ):
        super(SoftBALDStrategy, self).__init__(data_manager, model)
        assert temperature > 0, "temperature parameter should be greater than 0"
        self.T = torch.tensor(temperature)

    def active_learning_step(self, num_annotate: int) -> List[int]:
        self.model.train(self.l_loader, self.valid_loader)
        output = self.model(self.u_loader)
        scores = self.scoring_fn(x=output) / self.T
        scores = torch.softmax(scores, -1).numpy()
        num_annotate = min(num_annotate, len(self.u_indices))
        return np.random.choice(self.u_indices, size=num_annotate, replace=False, p=scores).tolist()
