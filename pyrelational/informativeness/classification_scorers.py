"""Collection of classification scorers for active learning strategies."""

import math

import torch
from torch import Tensor

from .abstract_scorers import AbstractClassificationScorer


class ClassificationLeastConfidence(AbstractClassificationScorer):
    """Least confidence classification scorer."""

    def __call__(self, prob_dist: Tensor, axis: int = -1) -> Tensor:
        """Compute the least confidence score.

        Returns the informativeness score of an array using least confidence
        sampling in a 0-1 range where 1 is the most uncertain.

        The least confidence uncertainty is the normalised difference between
        the most confident prediction and 100 percent confidence.
        """
        simple_least_conf, _ = torch.max(prob_dist, dim=axis)
        num_labels = prob_dist.size(axis)
        normalized_least_conf: Tensor = (1 - simple_least_conf) * (num_labels / (num_labels - 1))
        return normalized_least_conf


class MarginConfidence(AbstractClassificationScorer):
    """Margin confidence classification scorer."""

    def __call__(self, prob_dist: Tensor, axis: int = -1) -> Tensor:
        """Compute the margin confidence score.

        Returns the informativeness score of a probability distribution using
        margin of confidence sampling in a 0-1 range where 1 is the most uncertain
        The margin confidence uncertainty is the difference between the top two
        most confident predictions.
        """
        prob_dist = torch.sort(prob_dist, descending=True, dim=axis)[0]
        difference = prob_dist.select(axis, 0) - prob_dist.select(axis, 1)
        margin_conf: Tensor = 1 - difference
        return margin_conf


class RatioConfidence(AbstractClassificationScorer):
    """Ratio confidence classification scorer."""

    def __call__(self, prob_dist: Tensor, axis: int = -1) -> Tensor:
        """Compute the ratio confidence score.

        Returns the informativeness score of a probability distribution using
        ratio of confidence sampling in a 0-1 range where 1 is the most uncertain
        The ratio confidence uncertainty is the ratio between the top two most
        confident predictions.
        """
        prob_dist = torch.sort(prob_dist, descending=True, dim=axis)[0]
        ratio_conf: Tensor = prob_dist.select(axis, 0) / prob_dist.select(axis, 1)
        return ratio_conf


class Entropy(AbstractClassificationScorer):
    """Entropy classification scorer."""

    def __call__(self, prob_dist: Tensor, axis: int = -1) -> Tensor:
        r"""Compute the entropy score.

        Returns the informativeness score of a probability distribution using entropy.
        The entropy based uncertainty is defined as:
        :math:`- \frac{1}{\log(n)} \sum_{i}^{n} p_i \log (p_i)`
        """
        log_probs = prob_dist * torch.log2(prob_dist)
        raw_entropy = 0 - torch.sum(log_probs, dim=axis)
        normalised_entropy: Tensor = raw_entropy / math.log2(prob_dist.size(axis))
        return normalised_entropy


class ClassificationBald(AbstractClassificationScorer):
    """Entropy classification scorer."""

    def __init__(self) -> None:
        """Initialise the scorer."""
        super().__init__()
        self.entropy = Entropy()

    def __call__(self, prob_dist: Tensor, axis: int = -1) -> Tensor:
        r"""Compute the BALD score.

        Implementation of Bayesian Active Learning by Disagreement (BALD) for classification task
        `reference <https://arxiv.org/pdf/1112.5745.pdf>`__
        """
        return self.entropy(prob_dist, axis) - torch.mean(prob_dist * torch.log2(prob_dist), dim=axis)
