"""Unit tests for pyrelational.datasets.regression datasets
"""
from typing import Type, Union
from unittest import TestCase

from parameterized import parameterized

from pyrelational.datasets import (
    DiabetesDataset,
    SynthReg1,
    SynthReg2,
    UCIAirfoil,
    UCIConcrete,
    UCIEnergy,
    UCIPower,
    UCIWine,
    UCIYacht,
)


class TestRegressionBenchmarkDatasets(TestCase):
    """Class containing unit tests on benchmark regression datasets."""

    @parameterized.expand(
        [
            (SynthReg1, 500, 5),
            (SynthReg1, 1000, 3),
            (SynthReg2, 500, 5),
            (SynthReg2, 1000, 3),
        ]
    )
    def test_synthetic_datasets(
        self, dataset_class: Type[Union[SynthReg1, SynthReg2]], data_size: int, n_splits: int
    ) -> None:
        """Check attribute shapes of created dataset."""
        dataset = dataset_class(size=data_size, n_splits=n_splits)
        self.assertEqual(len(dataset), data_size)
        self.assertEqual(len(dataset.data_splits), n_splits)

    def test_DiabetesDataset(self) -> None:
        """Check attribute shapes of created dataset."""
        dataset = DiabetesDataset()
        self.assertEqual(len(dataset), 442)
        self.assertEqual(len(dataset.data_splits), 5)

    def test_UCIConcrete(self) -> None:
        """Check attribute shapes of created dataset."""
        dataset = UCIConcrete(data_dir="test_data/")
        self.assertEqual(len(dataset), 1030)
        self.assertEqual(dataset.x.shape[1], 8)
        self.assertEqual(len(dataset.data_splits), 5)

    def test_UCIEnergy(self) -> None:
        """Check attribute shapes of created dataset."""
        dataset = UCIEnergy(data_dir="test_data/")
        self.assertEqual(len(dataset), 768)
        self.assertEqual(dataset.x.shape[1], 9)
        self.assertEqual(len(dataset.data_splits), 5)

    def test_UCIPower(self) -> None:
        """Check attribute shapes of created dataset."""
        dataset = UCIPower(data_dir="test_data/")
        self.assertEqual(len(dataset), 9568)
        self.assertEqual(dataset.x.shape[1], 4)
        self.assertEqual(len(dataset.data_splits), 5)

    def test_UCIWine(self) -> None:
        """Check attribute shapes of created dataset."""
        dataset = UCIWine(data_dir="test_data/")
        self.assertEqual(len(dataset), 1598)
        self.assertEqual(dataset.x.shape[1], 11)
        self.assertEqual(len(dataset.data_splits), 5)

    def test_UCIYacht(self) -> None:
        """Check attribute shapes of created dataset."""
        dataset = UCIYacht(data_dir="test_data/")
        self.assertEqual(len(dataset), 306)
        self.assertEqual(dataset.x.shape[1], 6)
        self.assertEqual(len(dataset.data_splits), 5)

    def test_UCIAirfoil(self) -> None:
        """Check attribute shapes of created dataset."""
        dataset = UCIAirfoil(data_dir="test_data/")
        self.assertEqual(len(dataset), 1502)
        self.assertEqual(dataset.x.shape[1], 5)
        self.assertEqual(len(dataset.data_splits), 5)
