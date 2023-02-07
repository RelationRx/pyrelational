"""
This module contains methods for scoring samples based on model uncertainty in
regression tasks


Most of these functions are simple but giving them a name and implementation
in PyTorch is useful for defining the different active learning strategies
"""

from typing import Optional, Union, cast

import torch
from torch import Tensor
from torch.distributions import Distribution


def regression_mean_prediction(
    x: Optional[Union[Tensor, Distribution]] = None,
    mean: Optional[Tensor] = None,
    axis: int = 0,
) -> Tensor:
    """
    Returns mean score for each sample across repeats.
    Either x or mean should be provided as input.

    :param x:  pytorch tensor of repeat by scores (or scores by repeat) or pytorch Distribution
    :param mean:  pytorch tensor corresponding to the mean of a model's predictions for each sample
    :param axis: index of the axis along which the repeats are
    :return:  pytorch tensor of scores
    """
    _check_regression_informativeness_input(x, mean=mean)
    if mean is None:
        assert x is not None, "both x and mean are None, cannot compute."
        return _compute_mean(x, axis)
    return mean


def regression_least_confidence(
    x: Optional[Union[Tensor, Distribution]] = None,
    std: Optional[Tensor] = None,
    axis: int = 0,
) -> Tensor:
    """
    Implements least confidence scoring of based on input x returns std score for each sample across repeats.
    Either x or std should be provided as input.

    :param x:  pytorch tensor of repeat by scores (or scores by repeat) or pytorch Distribution
    :param std:  pytorch tensor corresponding to the standard deviation of a model's predictions for each sample
    :param axis: index of the axis along which the repeats are
    :return:  pytorch tensor of scores
    """
    _check_regression_informativeness_input(x, std=std)
    if std is None:
        assert x is not None, "both x and std are None, cannot compute."
        return _compute_std(x, axis)
    return std


def regression_expected_improvement(
    x: Optional[Union[Tensor, Distribution]] = None,
    mean: Optional[Tensor] = None,
    std: Optional[Tensor] = None,
    max_label: Union[float, Tensor] = 0.0,
    axis: int = 0,
    xi: float = 0.01,
) -> Tensor:
    """
    Implements expected improvement based on max_label in the currently available data
    (`reference <https://doi.org/10.1023/A:1008306431147>`__).
    Either x or mean and std should be provided as input.


    :param x:  pytorch tensor or pytorch Distribution
    :param mean:  pytorch tensor corresponding to a model's mean predictions for each sample
    :param std:  pytorch tensor corresponding to the standard deviation of a model's predictions for each sample
    :param max_label: max label in the labelled dataset
    :param axis: index of the axis along which the repeats are
    :param xi:  pytorch tensor or pytorch Distribution
    :return:  pytorch tensor of scores
    """
    _check_regression_informativeness_input(x, mean, std)
    if isinstance(x, Tensor):
        return torch.relu(x - max_label).mean(axis).flatten()
    else:
        mean = _compute_mean(x, axis) if mean is None else mean
        std = _compute_std(x, axis) if std is None else std
        Z = torch.relu(std) * (mean - max_label - xi)
        N = torch.distributions.Normal(0, 1)
        cdf, pdf = N.cdf(Z), torch.exp(N.log_prob(Z))
        return torch.relu((mean - max_label - xi) * cdf + std * pdf)


def regression_upper_confidence_bound(
    x: Optional[Union[Tensor, Distribution]] = None,
    mean: Optional[Tensor] = None,
    std: Optional[Tensor] = None,
    kappa: float = 1,
    axis: int = 0,
) -> Tensor:
    """
    Implements Upper Confidence Bound (UCB) scoring (`reference <https://doi.org/10.1023/A:1013689704352>`__)
    Either x or mean and std should be provided as input.

    :param x:  pytorch tensor or pytorch Distribution
    :param mean:  pytorch tensor corresponding to a model's mean predictions for each sample
    :param std:  pytorch tensor corresponding to the standard deviation of a model's predictions for each sample
    :param kappa: trade-off parameter between exploitation and exploration
    :param axis: index of the axis along which the repeats are
    :return:  pytorch tensor of scores
    """
    _check_regression_informativeness_input(x, mean, std)
    if mean is None:
        assert x is not None, "both x and mean are None, cannot compute."
        mean = _compute_mean(x, axis)
    if std is None:
        assert x is not None, "both x and std are None, cannot compute."
        std = _compute_std(x, axis)
    return mean + kappa * std


def regression_thompson_sampling(x: Tensor, axis: int = 0) -> Tensor:
    """
    Implements thompson sampling scoring (`reference <https://doi.org/10.1561/2200000070>`__).

    :param x:  pytorch tensor
    :param axis: index of the axis along which the repeats are
    :return:  pytorch tensor of scores
    """
    assert isinstance(x, Tensor), f"x input should be a torch Tensor, got {type(x)} instead."
    size = tuple(x.size(i) for i in range(x.ndim) if i != axis) + (1,)
    idx = torch.randint(high=x.size(axis), size=size)
    return x.gather(axis, idx).squeeze(-1)


def regression_bald(x: Tensor, axis: int = 0) -> Tensor:
    """
    Implementation of Bayesian Active Learning by Disagreement (BALD) for regression task
    (`reference <https://arxiv.org/pdf/1112.5745.pdf>`__)

    :param x:  pytorch Tensor
    :param axis: index of the axis along which the repeats are
    :return:  pytorch tensor of scores
    """
    assert isinstance(x, Tensor), f"x input should be a torch Tensor, got {type(x)} instead."
    x_mean = x.mean(axis, keepdim=True)
    x = (x - x_mean) ** 2
    return torch.log(1 + x.mean(axis)) / torch.tensor(2.0)


def _check_regression_informativeness_input(
    x: Optional[Union[Tensor, Distribution]] = None,
    mean: Optional[Tensor] = None,
    std: Optional[Tensor] = None,
) -> None:
    """
    Checks input to regression informativeness functions.

    :param x: predicted tensor or distribution
    :param mean: predicted mean
    :param std: predicted standard deviation
    """
    if x is None and mean is None and std is None:
        raise ValueError("Not all of x, mean, and std can be None.")

    if isinstance(x, Tensor):
        assert 2 <= x.ndim <= 3, "x input should be a 2D or 3D tensor"

    if isinstance(mean, Tensor):
        assert 1 <= mean.ndim <= 2, "mean input should be a 1D or 2D tensor"

    if isinstance(std, Tensor):
        assert 1 <= std.ndim <= 2, "std input should be a 1D or 2D tensor"


def _compute_mean(x: Optional[Union[Distribution, Tensor]], axis: int = 0) -> Tensor:
    """
    Compute mean of input.

    :param x: tensor or distribution
    :param axis: axis on which to take the mean (used when x is a Tensor)
    :return: mean vector
    """
    if isinstance(x, Tensor):
        return x.mean(axis)
    elif isinstance(x, Distribution):
        return cast(Tensor, x.mean)
    else:
        raise TypeError(f"Expected torch Tensor or Distribution, got {type(x)} instead.")


def _compute_std(x: Optional[Union[Distribution, Tensor]], axis: int = 0) -> Tensor:
    """
    Compute standard deviation of input.

    :param x: tensor or distribution
    :param axis: axis on which to take the standard deviation (used when x is a Tensor)
    :return: std vector
    """
    if isinstance(x, Tensor):
        return x.std(axis)
    elif isinstance(x, Distribution):
        return cast(Tensor, x.stddev)
    else:
        raise TypeError(f"Expected torch Tensor or Distribution, got {type(x)} instead.")
