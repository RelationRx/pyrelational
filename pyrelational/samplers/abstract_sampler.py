from abc import ABC, abstractmethod
from typing import Any, List

import torch
from torch import Tensor


class AbstractSampler(ABC):
    """_summary_"""

    @abstractmethod
    def __call__(self, scores: Tensor, indices: List[int], num_samples: int) -> List[int]:
        """_summary_

        :return: _description_
        """
        pass


class DeterministicSampler(AbstractSampler):
    """_summary_"""

    def __call__(self, scores: Tensor, indices: List[int], num_samples: int) -> List[int]:
        """_summary_

        :return: _description_
        """
        ixs = torch.argsort(scores, descending=True).tolist()
        return [indices[i] for i in ixs[:num_samples]]


class ProbabilisticSampler(AbstractSampler):
    """_summary_"""

    def __call__(self, scores: Tensor, indices: List[int], num_samples: int) -> List[int]:
        """_summary_

        :return: _description_
        """
        return [indices[i] for i in torch.multinomial(scores, num_samples, replacement=False).tolist()]
