from abc import ABC, abstractmethod
from typing import Any

from torch import Tensor

from .decorators import require_2d_tensor, require_probabilities


class AbstractScorer(ABC):
    """_summary_"""

    @abstractmethod
    def __call__(self, *args: Any, **kwargs: Any) -> Tensor:
        """_summary_

        :return: _description_
        """
        pass


class AbstractClassificationScorer(AbstractScorer):
    """_summary_"""

    @abstractmethod
    @require_probabilities
    @require_2d_tensor
    def __call__(self, prob_dist: Tensor, axis: int) -> Tensor:
        """_summary_

        :return: _description_
        """
        pass


class AbstractRegressionScorer(AbstractScorer):
    """_summary_"""

    @abstractmethod
    def __call__(self, *args: Any, **kwargs: Any) -> Tensor:
        """_summary_

        :return: _description_
        """
        pass
