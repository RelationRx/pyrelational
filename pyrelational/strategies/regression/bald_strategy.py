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

    def __init__(self):
        super(BALDStrategy, self).__init__()
        self.scoring_fn = regression_bald


class SoftBALDStrategy(BALDStrategy):
    """Implements soft BALD Strategy whereby unlabelled samples are queried based on mutual information score based on
    multiple estimator models. In contrast to Bald the query is drawn from unlabelled pool based on probabilities
    derived from bald scores instead of using an argmax operation"""

    def __init__(
        self,
        temperature: float = 0.5,
    ):
        super(SoftBALDStrategy, self).__init__()
        assert temperature > 0, "temperature parameter should be greater than 0"
        self.T = torch.tensor(temperature)

    def active_learning_step(
        self, num_annotate: int, data_manager: GenericDataManager, model: GenericModel
    ) -> List[int]:
        model.train(data_manager.get_labelled_loader(), data_manager.get_validation_loader())
        output = model(data_manager.get_unlabelled_loader())
        scores = self.scoring_fn(x=output) / self.T
        scores = torch.softmax(scores, -1).numpy()
        num_annotate = min(num_annotate, len(data_manager.u_indices))
        return np.random.choice(data_manager.u_indices, size=num_annotate, replace=False, p=scores).tolist()
