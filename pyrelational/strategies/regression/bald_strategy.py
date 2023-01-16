from typing import List

import numpy as np
import torch

from pyrelational.data import DataManager
from pyrelational.informativeness import regression_bald
from pyrelational.models import ModelManager
from pyrelational.strategies.regression.abstract_regression_strategy import (
    RegressionStrategy,
)


class BALDStrategy(RegressionStrategy):
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

    def __call__(self, num_annotate: int, data_manager: DataManager, model: ModelManager) -> List[int]:
        output = self.train_and_infer(data_manager=data_manager, model=model)
        scores = self.scoring_fn(x=output.squeeze(-1)) / self.T
        scores = torch.softmax(scores, -1).numpy()
        num_annotate = min(num_annotate, len(data_manager.u_indices))
        return np.random.choice(data_manager.u_indices, size=num_annotate, replace=False, p=scores).tolist()
