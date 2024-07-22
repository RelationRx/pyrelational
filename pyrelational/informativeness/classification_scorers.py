import math

import torch
from torch import Tensor

from .abstract_scorers import AbstractClassificationScorer
from .decorators import require_2d_tensor, require_probabilities


class ClassificationLeastConfidence(AbstractClassificationScorer):
    """_summary_"""

    @require_probabilities
    @require_2d_tensor
    def __call__(self, prob_dist: Tensor, axis: int = -1) -> Tensor:
        """_summary_

        :return: _description_
        """
        simple_least_conf, _ = torch.max(prob_dist, dim=axis)
        num_labels = prob_dist.size(axis)
        normalized_least_conf: Tensor = (1 - simple_least_conf) * (num_labels / (num_labels - 1))
        return normalized_least_conf


class MarginConfidence(AbstractClassificationScorer):
    """_summary_"""

    @require_probabilities
    @require_2d_tensor
    def __call__(self, prob_dist: Tensor, axis: int = -1) -> Tensor:
        """_summary_

        :return: _description_
        """
        prob_dist = torch.sort(prob_dist, descending=True, dim=axis)[0]
        difference = prob_dist.select(axis, 0) - prob_dist.select(axis, 1)
        margin_conf: Tensor = 1 - difference
        return margin_conf


class RatioConfidence(AbstractClassificationScorer):
    """_summary_"""

    @require_probabilities
    @require_2d_tensor
    def __call__(self, prob_dist: Tensor, axis: int = -1) -> Tensor:
        """_summary_

        :return: _description_
        """
        prob_dist = torch.sort(prob_dist, descending=True, dim=axis)[0]
        ratio_conf: Tensor = prob_dist.select(axis, 0) / prob_dist.select(axis, 1)
        return ratio_conf


class Entropy(AbstractClassificationScorer):
    """_summary_"""

    @require_probabilities
    @require_2d_tensor
    def __call__(self, prob_dist: Tensor, axis: int = -1) -> Tensor:
        """_summary_

        :return: _description_
        """
        log_probs = prob_dist * torch.log2(prob_dist)
        raw_entropy = 0 - torch.sum(log_probs, dim=axis)
        normalised_entropy: Tensor = raw_entropy / math.log2(prob_dist.size(axis))
        return normalised_entropy


class ClassificationBald(AbstractClassificationScorer):
    """_summary_"""

    def __init__(self):
        super().__init__()
        self.entropy = Entropy()

    @require_probabilities
    @require_2d_tensor
    def __call__(self, prob_dist: Tensor, axis: int = -1) -> Tensor:
        """_summary_

        :return: _description_
        """
        return self.entropy(prob_dist, axis) - torch.mean(prob_dist * torch.log2(prob_dist), dim=axis)
