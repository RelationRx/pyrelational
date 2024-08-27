"""Unit tests for uncertainty sampling methods."""

import math
from typing import Type, Union
from unittest import TestCase

import pytest
import torch
from parameterized import parameterized
from sklearn.base import ClusterMixin
from sklearn.cluster import AgglomerativeClustering
from torch.utils.data import DataLoader, TensorDataset

from pyrelational.informativeness import (
    AverageScorer,
    ClassificationBald,
    Entropy,
    ExpectedImprovement,
    LeastConfidence,
    MarginConfidence,
    RatioConfidence,
    RegressionBald,
    StandardDeviation,
    ThompsonSampling,
    UpperConfidenceBound,
)
from pyrelational.informativeness.abstract_scorers import AbstractRegressionScorer
from pyrelational.informativeness.task_agnostic_scorers import RelativeDistanceScorer
from pyrelational.strategies.task_agnostic.representative_sampling_strategy import (
    representative_sampling,
)


class TestInformativenessScorer(TestCase):
    """Class containing unit tests for informativeness measures."""

    @parameterized.expand(
        [
            (5, False, "KMeans"),
            (10, False, "AffinityPropagation"),
            (10, True, AgglomerativeClustering(n_clusters=10)),
        ]
    )
    def test_representative_sampling(
        self,
        num_annotations: int,
        use_dataloader: bool,
        clustering_method: Union[str, ClusterMixin],
    ) -> None:
        """Check returned query size."""
        query = torch.randn(100, 50)
        if use_dataloader:
            query = DataLoader(TensorDataset(query), batch_size=1)
        out = representative_sampling(query, num_annotate=num_annotations, clustering_method=clustering_method)
        self.assertGreaterEqual(num_annotations, len(out))

    @parameterized.expand(
        [
            (False, False),
            (True, False),
            (False, True),
            (True, True),
        ]
    )
    def test_relative_distance(self, reference_as_loader: bool, query_as_loader: bool) -> None:
        """Check output of relative distance function."""
        query = torch.tensor([[0, 0, 0], [1, 1, 1], [2, 2, 2]])
        if query_as_loader:
            query = DataLoader(TensorDataset(query), batch_size=1)
        reference = torch.tensor([[1, 1, 1], [2, 2, 2]])
        if reference_as_loader:
            reference = DataLoader(TensorDataset(reference), batch_size=1)

        scorer = RelativeDistanceScorer()
        out = scorer(query, reference)
        torch.testing.assert_close(out, torch.tensor([math.sqrt(3), 0, 0]), rtol=0, atol=1e-3)
        self.assertEqual(len(out), len(query))

    @parameterized.expand(
        [
            (AverageScorer,),
            (StandardDeviation,),
            (ThompsonSampling,),
            (RegressionBald,),
        ]
    )
    def test_regression_informativeness_with_tensor_input(
        self, informativeness: Type[AbstractRegressionScorer]
    ) -> None:
        """Check output dimension of informativeness functions."""
        a = torch.randn(25, 100)

        scorer = informativeness(axis=0)
        o = scorer(x=a)
        self.assertEqual(o.numel(), a.size(1))

        scorer = informativeness(axis=1)
        o = scorer(x=a)
        self.assertEqual(o.numel(), a.size(0))

    def test_regression_expected_improvement_with_tensor_input(self) -> None:
        """Check output dimension of expected improvement informativeness functions."""
        a = torch.randn(25, 100)

        scorer = ExpectedImprovement(axis=0)
        o = scorer(x=a, max_label=0.8)
        self.assertEqual(o.numel(), a.size(1))

        scorer = ExpectedImprovement(axis=1)
        o = scorer(x=a, max_label=0.8)
        self.assertEqual(o.numel(), a.size(0))

    def test_regression_ucb_with_tensor_input(self) -> None:
        """Check output dimension of expected improvement informativeness functions."""
        a = torch.randn(25, 100)

        scorer = UpperConfidenceBound(axis=0, kappa=0.25)
        o = scorer(x=a)
        self.assertEqual(o.numel(), a.size(1))

        scorer = UpperConfidenceBound(axis=1, kappa=0.25)
        o = scorer(x=a)
        self.assertEqual(o.numel(), a.size(0))

        a = torch.tensor([[0.1, 0.5, 0.5], [0.3, 1, 0.5]])
        scorer = UpperConfidenceBound(kappa=0.5)
        o = scorer(x=a)
        torch.testing.assert_close(o, torch.tensor([0.2707, 0.9268, 0.5000]), rtol=0, atol=1e-4)

    @parameterized.expand(
        [
            (AverageScorer,),
            (StandardDeviation,),
        ]
    )
    def test_regression_informativeness_with_distribution_input(
        self, informativeness: Type[AbstractRegressionScorer]
    ) -> None:
        """Check output dimension of informativeness functions when Distribution object is fed in."""
        a = torch.distributions.Normal(torch.randn(10), torch.rand(10))
        fn = informativeness()
        o = fn(x=a)
        self.assertEqual(o.numel(), a.loc.size(0))

    def test_regression_expected_improvement_with_distribution_input(self) -> None:
        """Check output dimension of expected improvement informativeness functions."""
        a = torch.distributions.Normal(torch.randn(10), torch.rand(10))
        scorer = ExpectedImprovement()
        o = scorer(x=a, max_label=0.8)
        self.assertEqual(o.numel(), a.loc.size(0))

    def test_regression_ucb_with_distribution_input(self) -> None:
        """Check output dimension of expected improvement informativeness functions."""
        a = torch.distributions.Normal(torch.randn(10), torch.rand(10))

        scorer = UpperConfidenceBound(kappa=0.25)
        o = scorer(x=a)
        self.assertEqual(o.numel(), a.loc.size(0))
        torch.testing.assert_close(o, a.loc + 0.25 * a.scale)

    def test_regression_greedy_with_mean_std_input(self) -> None:
        """Check output dimension of informativeness measures supporting mean/std input."""
        mean = torch.randn(10)
        o = AverageScorer()(mean=mean)
        self.assertEqual(o.numel(), mean.size(0))
        torch.testing.assert_close(mean, o)

    def test_regression_least_confidence_with_mean_std_input(self) -> None:
        """Check output dimension of informativeness measures supporting mean/std input."""
        std = torch.abs(torch.randn(10))
        o = StandardDeviation()(std=std)
        self.assertEqual(o.numel(), std.size(0))
        torch.testing.assert_close(o, std)

    def test_regression_expected_improvement_with_mean_std_input(self) -> None:
        """Check output dimension of informativeness measures supporting mean/std input."""
        mean, std = torch.randn(10), torch.abs(torch.randn(10))
        o = ExpectedImprovement()(mean=mean, std=std, max_label=0.8)
        self.assertEqual(o.numel(), mean.size(0))

    def test_regression_ucb_with_mean_std_input(self) -> None:
        """Check output dimension of informativeness measures supporting mean/std input."""
        mean, std = torch.randn(10), torch.abs(torch.randn(10))
        o = UpperConfidenceBound(kappa=0.25)(mean=mean, std=std)
        self.assertEqual(o.numel(), mean.size(0))
        torch.testing.assert_close(o, mean + 0.25 * std)

    def test_regression_input_check_fail_on_none(self) -> None:
        """Check that input check fails on empty inputs."""
        with pytest.raises(ValueError) as err:
            scorer = AverageScorer()
            scorer()
            self.assertEqual(str(err.value), "At least one of x, mean, or std must be provided.")

    def test_mean_std_compute_with_input_distribution(self) -> None:
        """Check the output of mean and std computation."""
        a = torch.distributions.Normal(torch.randn(10), torch.abs(torch.randn(10)))
        scorer = AverageScorer()
        m, s = scorer.compute_mean(a), scorer.compute_std(a)
        self.assertEqual(m.ndim, 1)
        self.assertEqual(s.ndim, 1)
        self.assertEqual(m.numel(), 10)
        self.assertEqual(s.numel(), 10)

    def test_mean_std_compute_with_input_tensor(self) -> None:
        """Check output of input check when provided with a 3D tensor."""
        a = torch.randn(25, 100)
        scorer = AverageScorer()
        m, s = scorer.compute_mean(a), scorer.compute_std(a)
        self.assertEqual(m.ndim, 1)
        self.assertEqual(s.ndim, 1)
        self.assertEqual(m.numel(), 100)
        self.assertEqual(s.numel(), 100)

    @parameterized.expand(
        [
            (torch.tensor([0.1, 0.5, 0.4]), pytest.approx(0.75, 0.01)),
            (torch.tensor([[0.5, 0.4, 0.1], [0.5, 0.4, 0.1]]), pytest.approx(0.75, 0.01)),
        ]
    )
    def test_least_confidence(self, inpt: torch.Tensor, expected: float) -> None:
        """Check correctness and output dimension."""
        out = LeastConfidence()(inpt)
        self.assertEqual(out, expected)
        self.assertEqual(out.numel(), inpt.ndim)

    def test_margin_confidence(self) -> None:
        """Check correctness."""
        prob_dist = torch.tensor([[0.1, 0.5, 0.4]])
        self.assertEqual(MarginConfidence()(prob_dist), pytest.approx(0.9, 0.01))

    def test_classification_bald(self) -> None:
        """Check correctness."""
        prob_dist = torch.tensor([[0.1, 0.5, 0.4], [0.3, 0.3, 0.4]]).unsqueeze(1)
        self.assertEqual(ClassificationBald()(prob_dist), pytest.approx(0.035, 0.01))

    def test_ratio_confidence(self) -> None:
        """Check correctness."""
        prob_dist = torch.tensor([[0.1, 0.5, 0.4]])
        self.assertEqual(RatioConfidence()(prob_dist), pytest.approx(0.8, 0.01))

    def test_entropy_based(self) -> None:
        """Check correctness."""
        prob_dist = torch.tensor([0.1, 0.5, 0.4])
        self.assertEqual(Entropy()(prob_dist), pytest.approx(0.8587, 0.01))
