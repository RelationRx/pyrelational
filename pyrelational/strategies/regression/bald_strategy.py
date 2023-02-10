from typing import Any, List

import numpy as np
import torch
from torch import Tensor

from pyrelational.data_managers import DataManager
from pyrelational.informativeness import regression_bald
from pyrelational.model_managers import ModelManager
from pyrelational.strategies.regression.abstract_regression_strategy import (
    RegressionStrategy,
)


class BALDStrategy(RegressionStrategy):
    """Implements BALD Strategy whereby unlabelled samples are queried based on mutual information score based on
    multiple estimator models."""

    def scoring_function(self, predictions: Tensor) -> Tensor:
        return regression_bald(predictions)


class SoftBALDStrategy(BALDStrategy):
    """Implements soft BALD Strategy whereby unlabelled samples are queried based on mutual information score based on
    multiple estimator models. In contrast to Bald the query is drawn from unlabelled pool based on probabilities
    derived from bald scores instead of using an argmax operation"""

    def __init__(
        self,
        temperature: float = 0.5,
    ):
        """
        :param temperature: parameter greater than 0 to divide scores before applying softmax.
            A lower temperature parameter will give a peakier distribution of probabilities
        """
        super(SoftBALDStrategy, self).__init__()
        assert temperature > 0, "temperature parameter should be greater than 0"
        self.T = torch.tensor(temperature)

    def __call__(
        self, num_annotate: int, data_manager: DataManager, model_manager: ModelManager[Any, Any]
    ) -> List[int]:
        """
        Call function which identifies samples which need to be labelled

        :param num_annotate: number of samples to annotate
        :param data_manager: A pyrelational data manager
            which keeps track of what has been labelled and creates data loaders for
            active learning
        :param model_manager: A pyrelational model manager
            which wraps a user defined ML model to handle instantiation, training, testing,
            as well as uncertainty quantification

        :return: list of indices to annotate
        """
        output = self.train_and_infer(data_manager=data_manager, model_manager=model_manager)
        scores = self.scoring_function(output).squeeze(-1)
        scores = torch.softmax(scores / self.T, -1).numpy()
        num_annotate = min(num_annotate, len(data_manager.u_indices))
        ret: List[int] = np.random.choice(data_manager.u_indices, size=num_annotate, replace=False, p=scores).tolist()
        return ret
