"""Unit tests for uci dataset downloader
"""
import os
import shutil
from unittest import TestCase

from parameterized import parameterized_class

from pyrelational.datasets import UCIDatasets


@parameterized_class([{"data_name": k} for k in UCIDatasets.datasets.keys()])
class TestUCIBenchmarkDatasets(TestCase):
    """Class containing unit tests on UCI benchmark datasets."""

    def setUp(self) -> None:
        """Set up class."""
        self.dataset = UCIDatasets(self.data_name, data_dir="test_data/", n_splits=10)

    def test_number_splits(self) -> None:
        """Check number of splits."""
        dataset = UCIDatasets("glass", data_dir="test_data/", n_splits=10)
        self.assertEqual(dataset.n_splits, 10)
        self.assertEqual(len(dataset.data_splits), 10)

    def test_split_size(self):
        """Check size of train and test splits."""
        split = self.dataset.get_split(train=True)
        self.assertEqual(len(split), len(self.dataset.data_splits[0][0]))

        split = self.dataset.get_split(train=False)
        self.assertEqual(len(split), len(self.dataset.data_splits[0][1]))

    def test_full_split_length(self) -> None:
        """Check full split length."""
        split = self.dataset.get_full_split()
        self.assertEqual(len(split), len(self.dataset.data))

    def test_get_simple_data(self) -> None:
        """Check size of returned simple dataset."""
        torch_dataset = self.dataset.get_simple_dataset()
        self.assertEqual(len(torch_dataset), len(self.dataset.data))

    def tearDown(self) -> None:
        """Tear down class."""
        if os.path.exists("test_data/"):
            shutil.rmtree("test_data")
