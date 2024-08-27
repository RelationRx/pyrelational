"""Collection of regression scorers for active learning strategies."""

from typing import Optional, Union

import torch
from torch import Tensor
from torch.distributions import Distribution

from pyrelational.informativeness.abstract_scorers import AbstractRegressionScorer
from pyrelational.informativeness.decorators import check_regression_input


class AverageScorer(AbstractRegressionScorer):
    """Scorer returning average prediction for each element."""

    @check_regression_input
    def __call__(self, x: Optional[Union[Tensor, Distribution]] = None, mean: Optional[Tensor] = None) -> Tensor:
        """Return mean score for each sample across repeats.

        Either x or mean should be provided as input.
        :param x:  pytorch tensor of repeat by scores (or scores by repeat) or pytorch Distribution
        :param mean:  pytorch tensor corresponding to the mean of a model's predictions for each sample
        :param axis: index of the axis along which the repeats are
        :return:  pytorch tensor of scores
        """
        if mean is None:
            assert x is not None, "both x and mean are None, cannot compute."
            return self.compute_mean(x)
        return mean


class StandardDeviation(AbstractRegressionScorer):
    """Scorer for least confidence in regression."""

    @check_regression_input
    def __call__(self, x: Optional[Union[Tensor, Distribution]] = None, std: Optional[Tensor] = None) -> Tensor:
        """Return standard deviation score for each sample across repeats.

        Either x or std should be provided as input.
        :param x:  pytorch tensor of repeat by scores (or scores by repeat) or pytorch Distribution
        :param std:  pytorch tensor corresponding to the standard deviation of a model's predictions for each sample
        :param axis: index of the axis along which the repeats are
        :return:  pytorch tensor of scores
        """
        if std is None:
            assert x is not None, "both x and std are None, cannot compute."
            return self.compute_std(x)
        return std


class ExpectedImprovement(AbstractRegressionScorer):
    r"""Scorer for expected improvement in regression (`reference <https://doi.org/10.1023/A:1008306431147>`__)."""

    def __init__(self, xi: float = 0.01, axis: int = 0) -> None:
        """Instantiate scorer."""
        super().__init__(axis=axis)
        self.xi = xi

    @check_regression_input
    def __call__(
        self,
        x: Optional[Union[Tensor, Distribution]] = None,
        mean: Optional[Tensor] = None,
        std: Optional[Tensor] = None,
        max_label: Union[float, Tensor] = 0.0,
    ) -> Tensor:
        """Return expected improvement score based on max label in the current data.

        Either x or mean and std should be provided as input.
        """
        if isinstance(x, Tensor):
            return torch.relu(x - max_label).mean(self.axis).flatten()
        else:
            if mean is None:
                assert x is not None, "both x and mean are None, cannot compute."
                mean = self.compute_mean(x)
            if std is None:
                assert x is not None, "both x and std are None, cannot compute."
                std = self.compute_std(x)
            return self._calculate_expected_improvement(mean, std)

    def _calculate_expected_improvement(
        self, mean: Tensor, std: Tensor, max_label: Union[float, Tensor] = 0.0
    ) -> Tensor:
        """Calculate expected improvement."""
        Z = (mean - max_label - self.xi) / std
        N = torch.distributions.Normal(0, 1)
        cdf, pdf = N.cdf(Z), torch.exp(N.log_prob(Z))
        return torch.relu((mean - max_label - self.xi) * cdf + std * pdf)


class UpperConfidenceBound(AbstractRegressionScorer):
    r"""Scorer for Upper Confidence Bound (UCB) in regression.

    `reference <https://doi.org/10.1023/A:1013689704352>`__
    """

    def __init__(self, kappa: float = 1, axis: int = 0) -> None:
        """Instantiate scorer.

        :param kappa: trade-off parameter between exploitation and exploration, defaults to 1
        :param axis: index of the axis along which the repeats are, defaults to 0
        """
        super().__init__(axis=axis)
        self.kappa = kappa

    @check_regression_input
    def __call__(
        self,
        x: Optional[Union[Tensor, Distribution]] = None,
        mean: Optional[Tensor] = None,
        std: Optional[Tensor] = None,
    ) -> Tensor:
        """Return UCB score based on mean and standard deviation.

        Either x or mean and std should be provided as input.
        :param x:  pytorch tensor or pytorch Distribution
        :param mean:  pytorch tensor corresponding to a model's mean predictions for each sample
        :param std:  pytorch tensor corresponding to the standard deviation of a model's predictions for each sample
        """
        if mean is None:
            assert x is not None, "both x and mean are None, cannot compute."
            mean = self.compute_mean(x)
        if std is None:
            assert x is not None, "both x and std are None, cannot compute."
            std = self.compute_std(x)
        return self._calculate_ucb(mean, std)

    def _calculate_ucb(self, mean: Tensor, std: Tensor) -> Tensor:
        """Calculate the Upper Confidence Bound."""
        return mean + self.kappa * std


class ThompsonSampling(AbstractRegressionScorer):
    r"""Scorer for Thompson Sampling in regression.

    `reference <https://doi.org/10.1561/2200000070>`__.
    """

    def __call__(self, x: Tensor) -> Tensor:
        """Return Thompson Sampling score by selecting random samples.

        :param x:  pytorch Tensor
        """
        size = tuple(x.size(i) for i in range(x.ndim) if i != self.axis) + (1,)
        idx = torch.randint(high=x.size(self.axis), size=size)
        return x.gather(self.axis, idx).squeeze(-1)


class RegressionBald(AbstractRegressionScorer):
    r"""Scorer for Bayesian Active Learning by Disagreement (BALD) in regression.

    `reference <https://arxiv.org/pdf/1112.5745.pdf>`__.
    Mathematically, Bald scorer is equivalent to the least confidence scorer.
    """

    def __call__(self, x: Tensor) -> Tensor:
        """Return BALD score based on disagreement among predictions."""
        x_mean = x.mean(self.axis, keepdim=True)
        x = (x - x_mean) ** 2
        return torch.log(1 + x.mean(self.axis)) / torch.tensor(2.0)
