from typing import Callable

import torch
from torch import Tensor


def require_probabilities(func: Callable[..., Tensor]) -> Callable[..., Tensor]:
    """
    Decorator to ensure that the input tensor is a probability distribution
    """

    def wrapper(prob_dist: Tensor, axis: int) -> Tensor:
        assert torch.allclose(
            prob_dist.sum(axis), torch.tensor(1.0)
        ), "input should be probability distributions along specified axis"
        return func(prob_dist, axis)

    return wrapper


def require_2d_tensor(func: Callable[..., Tensor]) -> Callable[..., Tensor]:
    """
    Decorator to ensure that the input tensor is a 2D tensor
    """

    def wrapper(x: Tensor, axis: int) -> Tensor:
        assert x.ndim == 2, "x input should be a 2D or 3D tensor"
        return func(x, axis)

    return wrapper
