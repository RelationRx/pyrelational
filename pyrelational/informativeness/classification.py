"""
This module contains methods for scoring samples based on model uncertainty in
classication tasks

This module contains functions for computing the informativeness values
of a given probability distribution (outputs of a model/mc-dropout
prediction, etc.)

"""

import math

import torch


def classification_least_confidence(prob_dist: torch.Tensor, axis: int = -1) -> torch.Tensor:
    r"""Returns the informativeness score of an array using least confidence
    sampling in a 0-1 range where 1 is the most uncertain

    The least confidence uncertainty is the normalised difference between
    the most confident prediction and 100 percent confidence

    Args:
        prob_dist (pytorch tensor): real number tensor whose elements add to 1.0
        sorted (bool): if the probability distribution is pre-sorted from
            largest to smallest
    """
    assert torch.allclose(
        prob_dist.sum(axis), torch.tensor(1.0)
    ), "input should be probability distributions along specified axis"

    simple_least_conf, _ = torch.max(prob_dist, dim=axis)
    num_labels = prob_dist.size(axis)
    normalized_least_conf = (1 - simple_least_conf) * (num_labels / (num_labels - 1))
    return normalized_least_conf


def classification_margin_confidence(prob_dist: torch.Tensor, axis: int = -1) -> torch.Tensor:
    r"""Returns the informativeness score of a probability distribution using
    margin of confidence sampling in a 0-1 range where 1 is the most uncertain
    The margin confidence uncertainty is the difference between the top two
    most confident predictions

    Args:
        prob_dist (pytorch tensor): real number tensor whose elements add to 1.0
        sorted (bool): if the probability distribution is pre-sorted from
            largest to smallest
    """
    assert torch.allclose(
        prob_dist.sum(axis), torch.tensor(1.0)
    ), "input should be probability distributions along specified axis"

    prob_dist, _ = torch.sort(prob_dist, descending=True, dim=axis)
    difference = prob_dist.select(axis, 0) - prob_dist.select(axis, 1)
    margin_conf = 1 - difference
    return margin_conf


def classification_ratio_confidence(prob_dist: torch.Tensor, axis: int = -1) -> torch.Tensor:
    r"""Returns the informativeness score of a probability distribution using
    ratio of confidence sampling in a 0-1 range where 1 is the most uncertain
    The ratio confidence uncertainty is the ratio between the top two most
    confident predictions

    Args:
        prob_dist (pytorch tensor): real number tensor whose elements add to 1.0
        sorted (bool): if the probability distribution is pre-sorted from
            largest to smallest
    """
    assert torch.allclose(
        prob_dist.sum(axis), torch.tensor(1.0)
    ), "input should be probability distributions along specified axis"

    prob_dist, _ = torch.sort(prob_dist, descending=True, dim=axis)  # sort probs so largest is first
    ratio_conf = prob_dist.select(axis, 1) / (prob_dist.select(axis, 0))  # ratio between top two props
    return ratio_conf


def classification_entropy(prob_dist: torch.Tensor, axis: int = -1) -> torch.Tensor:
    r"""Returns the informativeness score of a probability distribution
    using entropy

    The entropy based uncertainty is defined as

    :math:`- \frac{1}{\log(n)} \sum_{i}^{n} p_i \log (p_i)`

    Args:
        prob_dist (pytorch tensor): real number tensor whose elements add to 1.0
    """
    assert torch.allclose(
        prob_dist.sum(axis), torch.tensor(1.0)
    ), "input should be probability distributions along specified axis"

    log_probs = prob_dist * torch.log2(prob_dist)
    raw_entropy = 0 - torch.sum(log_probs, dim=axis)
    normalised_entropy = raw_entropy / math.log2(prob_dist.size(axis))

    return normalised_entropy


def classification_bald(prob_dist: torch.Tensor) -> torch.Tensor:
    """
    Implementation of Bayesian Active Learning by Disagreement (BALD) for classification task

    `reference <https://arxiv.org/pdf/1112.5745.pdf>`__
    :param x: 3D pytorch Tensor of shape n_estimators x n_samples x n_classes
    :return: 1D pytorch tensor of scores
    """

    assert torch.allclose(
        prob_dist.sum(-1), torch.tensor(1.0)
    ), "input should be probability distributions along specified axis"

    return classification_entropy(prob_dist.mean(0), -1) - classification_entropy(prob_dist, -1).mean(0)


def softmax(scores: torch.Tensor, base: float = math.e, axis: int = -1) -> torch.Tensor:
    """Returns softmax array for array of scores

    Converts a set of raw scores from a model (logits) into a
    probability distribution via softmax.

    The probability distribution will be a set of real numbers
    such that each is in the range 0-1.0 and the sum is 1.0.

    Assumes input is a pytorch tensor: tensor([1.0, 4.0, 2.0, 3.0])

    Args
        prediction (pytorch tensor) a pytorch tensor of any positive/negative real numbers.
        base (float) the base for the exponential (default e)
    """
    exps = base ** scores.float()  # exponential for each value in array
    sum_exps = torch.sum(exps, dim=axis, keepdim=True)  # sum of all exponentials
    prob_dist = exps / sum_exps  # normalize exponentials

    return prob_dist
