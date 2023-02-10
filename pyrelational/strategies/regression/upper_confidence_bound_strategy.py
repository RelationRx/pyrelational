from typing import Any, List

import torch

from pyrelational.data_managers import DataManager
from pyrelational.informativeness import regression_upper_confidence_bound
from pyrelational.model_managers import ModelManager
from pyrelational.strategies.abstract_strategy import Strategy


class UpperConfidenceBoundStrategy(Strategy):
    """Implements Upper Confidence Bound Strategy whereby unlabelled samples are scored and queried based on the
    UCB scorer"""

    def __init__(self, kappa: float = 1.0):
        """
        :param kappa: trade-off parameter between exploitation and exploration
        """
        super(UpperConfidenceBoundStrategy, self).__init__()
        self.kappa = kappa

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
        uncertainty = regression_upper_confidence_bound(x=output, kappa=self.kappa).squeeze(-1)
        ixs = torch.argsort(uncertainty, descending=True).tolist()
        return [data_manager.u_indices[i] for i in ixs[:num_annotate]]
