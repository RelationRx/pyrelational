"""Implement Expected Improvement Strategy for regression tasks."""

from typing import Any, List

import torch

from pyrelational.batch_mode_samplers import TopKSampler
from pyrelational.data_managers import DataManager
from pyrelational.informativeness import ExpectedImprovement
from pyrelational.model_managers import ModelManager
from pyrelational.strategies.abstract_strategy import Strategy


class ExpectedImprovementStrategy(Strategy):
    """Implement Expected Improvement Strategy.

    Unlabelled sample is scored based on the expected improvement scoring function.
    """

    scorer: ExpectedImprovement

    def __init__(self, xi: float = 0.01, axis: int = 0) -> None:
        """Initialize the strategy with the expected improvement scorer and a deterministic sampler for regression."""
        super().__init__(ExpectedImprovement(xi=xi, axis=axis), TopKSampler())

    def __call__(
        self, num_annotate: int, data_manager: DataManager, model_manager: ModelManager[Any, Any]
    ) -> List[int]:
        """
        Identify samples which need to be labelled.

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
        uncertainty = self.scorer(output, max_label=max_label)
        return self.sampler(uncertainty, data_manager.u_indices, num_annotate)
