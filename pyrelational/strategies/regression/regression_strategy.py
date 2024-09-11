"""Regression strategy class implementing __call__ logic."""

from typing import Any, List

from pyrelational.data_managers import DataManager
from pyrelational.model_managers import ModelManager
from pyrelational.strategies.abstract_strategy import Strategy


class RegressionStrategy(Strategy):
    """A base active learning strategy class for regression."""

    def __call__(
        self, num_annotate: int, data_manager: DataManager, model_manager: ModelManager[Any, Any]
    ) -> List[int]:
        """
        Identify samples for labelling based on user defined scoring and sampling function.

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
        if output.shape[0] == 1:
            scores = self.scorer(output)
        else:
            scores = self.scorer(output).squeeze(-1)
        return self.sampler(scores, data_manager.u_indices, num_annotate)
