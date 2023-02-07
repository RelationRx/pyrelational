"""Unit tests for uncertainty sampling methods
"""
import math
from typing import Union
from unittest import TestCase

import pytest
import torch
from parameterized import parameterized
from sklearn.base import ClusterMixin
from sklearn.cluster import AgglomerativeClustering
from torch.utils.data import DataLoader, TensorDataset

import pyrelational.informativeness.regression as runc
from pyrelational.informativeness import (
    classification_bald,
    classification_entropy,
    classification_least_confidence,
    classification_margin_confidence,
    classification_ratio_confidence,
    relative_distance,
    representative_sampling,
    softmax,
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
        query = torch.tensor([[0, 0, 0], [1, 1, 1], [2, 2, 2]])
        if query_as_loader:
            query = DataLoader(TensorDataset(query), batch_size=1)
        reference = torch.tensor([[1, 1, 1], [2, 2, 2]])
        if reference_as_loader:
            reference = DataLoader(TensorDataset(reference), batch_size=1)

        out = relative_distance(query, reference)
        torch.testing.assert_close(out, torch.tensor([math.sqrt(3), 0, 0]), rtol=0, atol=1e-3)
        self.assertEqual(len(out), len(query))

    @parameterized.expand(
        [
            ("regression_mean_prediction",),
            ("regression_thompson_sampling",),
            ("regression_least_confidence",),
            ("regression_bald",),
        ]
    )
    def test_regression_informativeness_with_tensor_input(self, informativeness: str) -> None:
        """Check output dimension of informativeness functions."""
        a = torch.randn(25, 100)
        fn = getattr(runc, informativeness)
        o = fn(x=a, axis=0)
        self.assertEqual(o.numel(), a.size(1))
        o = fn(x=a, axis=1)
        self.assertEqual(o.numel(), a.size(0))

    def test_regression_expected_improvement_with_tensor_input(self) -> None:
        """Check output dimension of expected improvement informativeness functions."""
        a = torch.randn(25, 100)
        o = runc.regression_expected_improvement(x=a, max_label=0.8, axis=0)
        self.assertEqual(o.numel(), a.size(1))
        o = runc.regression_expected_improvement(x=a, max_label=0.8, axis=1)
        self.assertEqual(o.numel(), a.size(0))

    def test_regression_ucb_with_tensor_input(self) -> None:
        """Check output dimension of expected improvement informativeness functions."""
        a = torch.randn(25, 100)
        o = runc.regression_upper_confidence_bound(x=a, kappa=0.25, axis=0)
        self.assertEqual(o.numel(), a.size(1))
        o = runc.regression_upper_confidence_bound(x=a, kappa=0.25, axis=1)
        self.assertEqual(o.numel(), a.size(0))

        a = torch.tensor([[0.1, 0.5, 0.5], [0.3, 1, 0.5]])
        o = runc.regression_upper_confidence_bound(x=a, kappa=0.5)
        torch.testing.assert_close(o, torch.tensor([0.2707, 0.9268, 0.5000]), rtol=0, atol=1e-4)

    @parameterized.expand(
        [
            ("regression_mean_prediction",),
            ("regression_least_confidence",),
        ]
    )
    def test_regression_informativeness_with_distribution_input(self, informativeness: str) -> None:
        a = torch.distributions.Normal(torch.randn(10), torch.rand(10))
        fn = getattr(runc, informativeness)
        o = fn(x=a)
        self.assertEqual(o.numel(), a.loc.size(0))

    def test_regression_expected_improvement_with_distribution_input(self) -> None:
        """Check output dimension of expected improvement informativeness functions."""
        a = torch.distributions.Normal(torch.randn(10), torch.rand(10))
        o = runc.regression_expected_improvement(x=a, max_label=0.8)
        self.assertEqual(o.numel(), a.loc.size(0))

    def test_regression_ucb_with_distribution_input(self) -> None:
        """Check output dimension of expected improvement informativeness functions."""
        a = torch.distributions.Normal(torch.randn(10), torch.rand(10))
        o = runc.regression_upper_confidence_bound(x=a, kappa=0.25)
        self.assertEqual(o.numel(), a.loc.size(0))
        torch.testing.assert_close(o, a.loc + 0.25 * a.scale)

    def test_regression_greedy_with_mean_std_input(self) -> None:
        """Check output dimension of informativeness measures supporting mean/std input."""
        mean = torch.randn(10)
        o = runc.regression_mean_prediction(mean=mean)
        self.assertEqual(o.numel(), mean.size(0))
        torch.testing.assert_close(mean, o)

    def test_regression_least_confidence_with_mean_std_input(self) -> None:
        """Check output dimension of informativeness measures supporting mean/std input."""
        std = torch.abs(torch.randn(10))
        o = runc.regression_least_confidence(std=std)
        self.assertEqual(o.numel(), std.size(0))
        torch.testing.assert_close(o, std)

    def test_regression_expected_improvement_with_mean_std_input(self) -> None:
        """Check output dimension of informativeness measures supporting mean/std input."""
        mean, std = torch.randn(10), torch.abs(torch.randn(10))
        o = runc.regression_expected_improvement(mean=mean, std=std, max_label=0.8)
        self.assertEqual(o.numel(), mean.size(0))

    def test_regression_ucb_with_mean_std_input(self) -> None:
        """Check output dimension of informativeness measures supporting mean/std input."""
        mean, std = torch.randn(10), torch.abs(torch.randn(10))
        o = runc.regression_upper_confidence_bound(mean=mean, std=std, kappa=0.25)
        self.assertEqual(o.numel(), mean.size(0))
        torch.testing.assert_close(o, mean + 0.25 * std)

    def test_regression_input_check_fail_on_none(self) -> None:
        """Check that input check fails on empty inputs."""
        with pytest.raises(ValueError) as err:
            runc._check_regression_informativeness_input()
            self.assertEqual(str(err.value), "Not all of x, mean, and std can be None.")

    def test_regression_input_check_with_distribution_input(self) -> None:
        """Check that the check returns no x tensor and correct shapes."""
        a = torch.distributions.Normal(torch.randn(10), torch.abs(torch.randn(10)))
        m, s = runc._compute_mean(a), runc._compute_std(a)
        self.assertEqual(m.numel(), 10)
        self.assertEqual(s.numel(), 10)

    def test_regression_input_check_fail_with_2D_distribution_input(self) -> None:
        """Check that the check returns no x tensor and correct shapes."""
        a = torch.distributions.Normal(torch.randn(10, 2), torch.abs(torch.randn(10, 2)))
        with pytest.raises(AssertionError) as err:
            runc._check_regression_informativeness_input(a)
            self.assertEqual(str(err.value), "distribution input should be 1D")

    def test_mean_std_computation(self) -> None:
        """Check output of input check when provided with a 3D tensor."""
        a = torch.randn(25, 100)
        m, s = runc._compute_mean(a), runc._compute_std(a)
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
        out = classification_least_confidence(inpt)
        self.assertEqual(out, expected)
        self.assertEqual(out.numel(), inpt.ndim)

    def test_margin_confidence(self) -> None:
        """Check correctness."""
        prob_dist = torch.tensor([[0.1, 0.5, 0.4]])
        self.assertEqual(classification_margin_confidence(prob_dist), pytest.approx(0.9, 0.01))

    def test_classificaiton_bald(self) -> None:
        """Check correctness."""
        prob_dist = torch.tensor([[0.1, 0.5, 0.4], [0.3, 0.3, 0.4]]).unsqueeze(1)
        self.assertEqual(classification_bald(prob_dist), pytest.approx(0.035, 0.01))

    def test_ratio_confidence(self) -> None:
        """Check correctness."""
        prob_dist = torch.tensor([[0.1, 0.5, 0.4]])
        self.assertEqual(classification_ratio_confidence(prob_dist), pytest.approx(0.8, 0.01))

    def test_entropy_based(self) -> None:
        """Check correctness."""
        prob_dist = torch.tensor([0.1, 0.5, 0.4])
        self.assertEqual(classification_entropy(prob_dist), pytest.approx(0.8587, 0.01))

    def test_softmax(self) -> None:
        """Check correctness."""
        scores = torch.tensor([1.0, 4.0, 2.0, 3.0])
        answer = torch.tensor([0.0321, 0.6439, 0.0871, 0.2369])
        torch.testing.assert_close(softmax(scores, base=math.e), answer, rtol=1e-3, atol=1e-4)
        self.assertEqual(sum(softmax(scores, base=math.e)).item(), pytest.approx(1, 0.1))
