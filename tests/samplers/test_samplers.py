"""Tests for the samplers module.""" ""
from unittest import TestCase

import torch

from pyrelational.batch_mode_samplers import ProbabilisticSampler, TopKSampler


class TestSamplers(TestCase):
    """Collection of tests for samplers."""

    def test_deterministic_sampler(self) -> None:
        """Test deterministic sampler."""
        sampler = TopKSampler()
        query = sampler(torch.tensor([0.1, 3.0, 2.1]), [1, 2, 3], 1)
        self.assertEqual(len(query), 1)
        self.assertEqual(query, [2])

    def test_probabilistic_sampler(self) -> None:
        """Test probabilistic sampler."""
        sampler = ProbabilisticSampler()
        query = sampler(torch.tensor([0.1, 0.2, 0.7]), [1, 2, 3], 1)
        self.assertEqual(len(query), 1)
