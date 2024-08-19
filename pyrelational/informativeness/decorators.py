"""Decorators for checking input shapes and types for scorers."""

from functools import wraps
from typing import TYPE_CHECKING, Any, Callable, Optional, Union

import torch
from torch import Tensor
from torch.distributions import Distribution

if TYPE_CHECKING:
    from pyrelational.informativeness.abstract_scorers import (
        AbstractClassificationScorer,
        AbstractRegressionScorer,
    )


def require_probabilities(func: Callable[..., Tensor]) -> Callable[..., Tensor]:
    """Ensure that the input tensor is a probability distribution."""

    @wraps(func)
    def wrapper(self: "AbstractClassificationScorer", prob_dist: Tensor) -> Tensor:
        """Check the input tensor sums to 1 along axis."""
        assert torch.allclose(
            prob_dist.sum(self.axis), torch.tensor(1.0)
        ), "input should be probability distributions along specified axis"
        return func(self, prob_dist)

    return wrapper


def check_regression_input(func: Callable[..., Tensor]) -> Callable[..., Tensor]:
    """Check inputs for regression scoring functions."""

    @wraps(func)
    def wrapper(
        self: "AbstractRegressionScorer", x: Optional[Union[Tensor, Distribution]] = None, **kwargs: Any
    ) -> Tensor:
        """Check shapes of input tensors."""
        mean = kwargs.get("mean", None)
        std = kwargs.get("std", None)
        if x is None and mean is None and std is None:
            raise ValueError("At least one of x, mean, or std must be provided.")

        if isinstance(x, Tensor):
            assert 2 <= x.ndim <= 3, "x input should be a 2D or 3D tensor"

        if isinstance(mean, Tensor):
            assert 1 <= mean.ndim <= 2, "mean input should be a 1D or 2D tensor"

        if isinstance(std, Tensor):
            assert 1 <= std.ndim <= 2, "std input should be a 1D or 2D tensor"

        return func(self, x, **kwargs)

    return wrapper
