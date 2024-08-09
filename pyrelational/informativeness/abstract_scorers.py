"""Abstract scorer classes to define the interface for informativeness scorers."""

from abc import ABC, abstractmethod
from typing import Any

from torch import Tensor

from .decorators import require_2d_tensor, require_probabilities


class AbstractScorer(ABC):
    """Abstract scorer class."""

    @abstractmethod
    def __call__(self, *args: Any, **kwargs: Any) -> Tensor:
        """Score the input data.

        This method should be implemented by the subclasses.
        :return: scores associated with the input data.
        """
        pass


class DecoratedClassificationScorerMeta(type):
    """Metaclass for classification scorers."""

    def __new__(cls, name: str, bases: tuple[Any, ...], dct: dict[str, Any]) -> "DecoratedClassificationScorerMeta":
        """Decorate the `__call__` method with the `require_2d_tensor` and `require_probabilities` decorators."""
        for key, value in dct.items():
            if callable(value) and key == "__call__":
                dct[key] = require_probabilities(require_2d_tensor(value))
        return super().__new__(cls, name, bases, dct)


class AbstractClassificationScorer(metaclass=DecoratedClassificationScorerMeta):
    """Abstract classification scorer class."""

    @abstractmethod
    def __call__(self, prob_dist: Tensor, axis: int) -> Tensor:
        """Score the input data.

        This method should be implemented by the subclasses conserving the decorators and signature.
        :return: scores associated with the input data.
        """
        pass
