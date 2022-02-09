"""
This module contains methods for scoring samples based on model uncertainty in
regression tasks


Most of these functions are simple but giving them a name and implementation
in PyTorch is useful for defining the different active learning strategies
"""

from typing import Optional, Union

import torch
from torch.distributions import Distribution


def regression_greedy_score(
    x: Optional[Union[torch.Tensor, Distribution]] = None,
    mean: Optional[torch.Tensor] = None,
    axis: int = 0,
) -> torch.Tensor:
    """
    Implements greedy scoring that returns mean score for each sample across repeats.
    Either x or mean should be provided as input.

    :param x: 2D pytorch tensor of repeat by scores (or scores by repeat) or pytorch Distribution
    :param std: 1D pytorch tensor corresponding to the standard deviation of a model's predictions for each sample
    :param axis: index of the axis along which the repeats are
    :return: 1D pytorch tensor of scores
    """
    _, mean, _ = _check_regression_informativeness_input(x, mean=mean, axis=axis)
    return mean


def regression_least_confidence(
    x: Optional[Union[torch.Tensor, Distribution]] = None,
    std: Optional[torch.Tensor] = None,
    axis: int = 0,
) -> torch.Tensor:
    """
    Implements least confidence scoring of based on input x returns std score for each sample across repeats.
    Either x or std should be provided as input.

    :param x: 2D pytorch tensor of repeat by scores (or scores by repeat) or pytorch Distribution
    :param std: 1D pytorch tensor corresponding to the standard deviation of a model's predictions for each sample
    :param axis: index of the axis along which the repeats are
    :return: 1D pytorch tensor of scores
    """
    _, _, std = _check_regression_informativeness_input(x, std=std, axis=axis)
    return std


def regression_expected_improvement(
    x: Optional[Union[torch.Tensor, Distribution]] = None,
    mean: Optional[torch.Tensor] = None,
    std: Optional[torch.Tensor] = None,
    max_label: float = 0.0,
    axis: int = 0,
    xi: float = 0.01,
) -> torch.Tensor:
    """
    Implements expected improvement based on max_label in the currently available data
    (`reference <https://doi.org/10.1023/A:1008306431147>`__).
    Either x or mean and std should be provided as input.


    :param x: 2D pytorch tensor or pytorch Distribution
    :param mean: 1D pytorch tensor corresponding to a model's mean predictions for each sample
    :param std: 1D pytorch tensor corresponding to the standard deviation of a model's predictions for each sample
    :param max_label: max label in the labelled dataset
    :param axis: index of the axis along which the repeats are
    :param xi: 2D pytorch tensor or pytorch Distribution
    :return: 1D pytorch tensor of scores
    """
    x, mean, std = _check_regression_informativeness_input(x, mean, std, axis=axis)
    if isinstance(x, torch.Tensor):
        return torch.relu(x - max_label).mean(axis).flatten()
    else:
        Z = torch.relu(std) * (mean - max_label - xi)
        N = torch.distributions.Normal(0, 1)
        cdf, pdf = N.cdf(Z), torch.exp(N.log_prob(Z))
        return torch.relu((mean - max_label - xi) * cdf + std * pdf)


def regression_upper_confidence_bound(
    x: Optional[Union[torch.Tensor, Distribution]] = None,
    mean: Optional[torch.Tensor] = None,
    std: Optional[torch.Tensor] = None,
    kappa: float = 1,
    axis: int = 0,
) -> torch.Tensor:
    """
    Implements Upper Confidence Bound (UCB) scoring (`reference <https://doi.org/10.1023/A:1013689704352>`__)
    Either x or mean and std should be provided as input.

    :param x: 2D pytorch tensor or pytorch Distribution
    :param mean: 1D pytorch tensor corresponding to a model's mean predictions for each sample
    :param std: 1D pytorch tensor corresponding to the standard deviation of a model's predictions for each sample
    :param kappa: trade-off parameter between exploitation and exploration
    :param axis: index of the axis along which the repeats are
    :return: 1D pytorch tensor of scores
    """

    _, mean, std = _check_regression_informativeness_input(x, mean, std, axis=axis)
    return mean + kappa * std


def regression_thompson_sampling(x: torch.Tensor, axis: int = 0) -> torch.Tensor:
    """
    Implements thompson sampling scoring (`reference <https://doi.org/10.1561/2200000070>`__).

    :param x: 2D pytorch tensor
    :param axis: index of the axis along which the repeats are
    :return: 1D pytorch tensor of scores
    """
    x, _, _ = _check_regression_informativeness_input(x, axis=axis)
    other_axis = (axis - 1) % 2
    idx = torch.randint(high=x.size(axis), size=(x.size(other_axis), 1))
    return x.gather(axis, idx).flatten()


def regression_bald(x: torch.Tensor, axis: int = 0) -> torch.Tensor:
    """
    Implementation of Bayesian Active Learning by Disagreement (BALD) for regression task
    (`reference <https://arxiv.org/pdf/1112.5745.pdf>`__)

    :param x: 2D pytorch Tensor
    :param axis: index of the axis along which the repeats are
    :return: 1D pytorch tensor of scores
    """
    x, _, _ = _check_regression_informativeness_input(x, axis=axis)
    x_mean = x.mean(axis, keepdim=True)
    x = (x - x_mean) ** 2
    return torch.log(1 + x.mean(axis)) / torch.tensor(2.0)


def _check_regression_informativeness_input(x=None, mean=None, std=None, axis=0):
    if x is None and mean is None and std is None:
        raise ValueError("Not all of x, mean, and std can be None.")

    if isinstance(x, torch.Tensor):
        x = x.squeeze()
        assert x.ndim == 2, "x input should be a 2D tensor"
        return x, x.mean(axis), x.std(axis)

    if isinstance(mean, torch.Tensor) and isinstance(std, torch.Tensor):
        mean, std = mean.squeeze(), std.squeeze()
        assert mean.ndim == 1, "mean input should be a 1D tensor"
        assert std.ndim == 1, "std input should be a 1D tensor"
        return None, mean, std
    elif isinstance(mean, torch.Tensor):
        mean = mean.squeeze()
        assert mean.ndim == 1, "mean input should be a 1D tensor"
        return None, mean, None
    elif isinstance(std, torch.Tensor):
        std = std.squeeze()
        assert std.ndim == 1, "std input should be a 1D tensor"
        return None, None, std

    if isinstance(x, Distribution):
        mean, std = x.mean.squeeze(), x.stddev.squeeze()
        assert mean.ndim == 1, "distribution input should be 1D"
        return None, mean, std
