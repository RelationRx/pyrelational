"""Unit tests for benchmark datamanager functions
"""
from unittest import TestCase

from pyrelational.datasets import BreastCancerDataset, DiabetesDataset
from pyrelational.datasets.benchmark_datamanager import (
    create_classification_cold_start,
    create_regression_cold_start,
    create_warm_start,
)


class TestBenchmarkDataManager(TestCase):
    """Class containing unit tests for benchmark datamanager creation."""

    def test_create_warm_start_classification(self) -> None:
        """Check shape correctness of dataset."""
        dataset = BreastCancerDataset()
        train_indices = list(dataset.data_splits[0][0])
        test_indices = list(dataset.data_splits[0][1])
        dm = create_warm_start(dataset, train_indices=train_indices, test_indices=test_indices)
        self.assertEqual(len(dm), 569)

    def test_create_warm_start_regression(self) -> None:
        """Check shape correctness of dataset."""
        dataset = DiabetesDataset()
        train_indices = list(dataset.data_splits[0][0])
        test_indices = list(dataset.data_splits[0][1])
        dm = create_warm_start(dataset, train_indices=train_indices, test_indices=test_indices)
        self.assertEqual(len(dm), 442)

    def test_create_classification_cold_start(self) -> None:
        """Check shape correctness of dataset."""
        dataset = BreastCancerDataset()
        train_indices = list(dataset.data_splits[0][0])
        test_indices = list(dataset.data_splits[0][1])
        dm = create_classification_cold_start(dataset, train_indices=train_indices, test_indices=test_indices)
        self.assertEqual(len(dm), 569)
        self.assertEqual(len(dm.l_indices), 2)

    def test_create_regression_cold_start(self) -> None:
        """Check shape correctness of dataset."""
        dataset = DiabetesDataset()
        train_indices = list(dataset.data_splits[0][0])
        test_indices = list(dataset.data_splits[0][1])
        dm = create_regression_cold_start(dataset, train_indices=train_indices, test_indices=test_indices)
        self.assertEqual(len(dm), 442)
        self.assertEqual(len(dm.l_indices), 2)
