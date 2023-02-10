from typing import Any, List

import torch

from pyrelational.data_managers import DataManager
from pyrelational.informativeness import regression_expected_improvement
from pyrelational.model_managers import ModelManager
from pyrelational.strategies.abstract_strategy import Strategy


class ExpectedImprovementStrategy(Strategy):
    """Implement Expected Improvement Strategy whereby each unlabelled sample is scored based on the
    expected improvement scoring function. The top samples according to this score are selected at each step"""

    def __init__(self) -> None:
        super(ExpectedImprovementStrategy, self).__init__()

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
        max_label = torch.max(data_manager.get_sample_labels(data_manager.l_indices))
        uncertainty = regression_expected_improvement(x=output, max_label=max_label).squeeze(-1)
        ixs = torch.argsort(uncertainty, descending=True).tolist()
        return [data_manager.u_indices[i] for i in ixs[:num_annotate]]
