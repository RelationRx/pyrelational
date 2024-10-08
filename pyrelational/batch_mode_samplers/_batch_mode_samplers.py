"""Collection of samplers for active learning strategies."""

from abc import ABC, abstractmethod
from typing import List

import torch
from torch import Tensor


class BatchModeSampler(ABC):
    """Abstract sampler class."""

    @abstractmethod
    def __call__(self, scores: Tensor, indices: List[int], num_samples: int) -> List[int]:
        """Sample a subset of indices based on the scores.

        This method should be implemented by the subclasses.
        :return: List of sampled indices.
        """
        pass


class TopKSampler(BatchModeSampler):
    """Deterministic sampler based on the top-k scores."""

    def __call__(self, scores: Tensor, indices: List[int], num_samples: int) -> List[int]:
        """Sample the top-k indices based on the scores.

        :return: List of sampled indices.
        """
        ixs = torch.argsort(scores, descending=True).tolist()
        return [indices[i] for i in ixs[:num_samples]]


class ProbabilisticSampler(BatchModeSampler):
    """Probabilistic sampler based on the scores."""

    def __call__(self, scores: Tensor, indices: List[int], num_samples: int) -> List[int]:
        """Sample a subset of indices deriving a distribution from the scores.

        :return: List of sampled indices.
        """
        num_samples = min(num_samples, len(indices))
        return [indices[i] for i in torch.multinomial(scores, num_samples, replacement=False).tolist()]
