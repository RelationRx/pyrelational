"""Abstract scorer classes to define the interface for informativeness scorers."""

from abc import ABC, abstractmethod
from typing import Any, Union

from torch import Tensor
from torch.distributions import Distribution

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

    def __init__(self, axis: int = -1) -> None:
        """Instantiate scorer."""
        super().__init__()
        self.axis = axis

    @abstractmethod
    def __call__(self, prob_dist: Tensor) -> Tensor:
        """Score the input data.

        This method should be implemented by the subclasses conserving the decorators and signature.
        :return: scores associated with the input data.
        """
        pass


class AbstractRegressionScorer(AbstractScorer):
    """Abstract base class for all regression scorers."""

    def __init__(self, axis: int = 0) -> None:
        """Instantiate scorer."""
        super().__init__()
        self.axis = axis

    def compute_mean(self, x: Union[Tensor, Distribution]) -> Tensor:
        """Compute the mean of the input tensor or distribution."""
        if isinstance(x, Tensor):
            return x.mean(self.axis)
        elif isinstance(x, Distribution):
            return x.mean
        else:
            raise TypeError(f"Expected torch Tensor or Distribution, got {type(x)} instead.")

    def compute_std(self, x: Union[Tensor, Distribution]) -> Tensor:
        """Compute the standard deviation of the input tensor or distribution."""
        if isinstance(x, Tensor):
            return x.std(self.axis)
        elif isinstance(x, Distribution):
            return x.stddev
        else:
            raise TypeError(f"Expected torch Tensor or Distribution, got {type(x)} instead.")
