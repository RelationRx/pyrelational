"""Unit tests for classification datasets
"""
from typing import Type, Union
from unittest import TestCase

from parameterized import parameterized

from pyrelational.datasets import (
    BreastCancerDataset,
    Checkerboard2x2Dataset,
    Checkerboard4x4Dataset,
    CreditCardDataset,
    DigitDataset,
    FashionMNIST,
    GaussianCloudsDataset,
    StriatumDataset,
    SynthClass1,
    SynthClass2,
    SynthClass3,
    UCIGlass,
    UCIParkinsons,
    UCISeeds,
)


class TestClassificationBenchmarkDatasets(TestCase):
    """Class containing unit tests on benchmark classification datasets."""

    @parameterized.expand(
        [
            (SynthClass1, 500, 5),
            (SynthClass1, 1000, 3),
            (SynthClass2, 500, 5),
            (SynthClass2, 1000, 3),
            (SynthClass3, 500, 5),
            (SynthClass3, 1000, 3),
        ]
    )
    def test_synthetic_datasets(
        self, dataset_class: Type[Union[SynthClass1, SynthClass2, SynthClass3]], data_size: int, n_splits: int
    ) -> None:
        """Check attribute shapes of created dataset."""
        dataset = dataset_class(size=data_size, n_splits=n_splits)
        self.assertEqual(len(dataset), data_size)
        self.assertEqual(len(dataset.data_splits), n_splits)

    def test_BreastCancerDataset(self) -> None:
        """Check attribute shapes of created dataset."""
        dataset = BreastCancerDataset()
        self.assertEqual(len(dataset), 569)
        self.assertEqual(dataset.x.shape[1], 30)
        self.assertEqual(len(dataset.data_splits), 5)

    def test_DigitDataset(self) -> None:
        """Check attribute shapes of created dataset."""
        dataset = DigitDataset()
        self.assertEqual(len(dataset), 1797)
        self.assertEqual(dataset.x.shape[1], 64)
        self.assertEqual(len(dataset.data_splits), 5)

    def test_FashionMNIST(self) -> None:
        """Check attribute shapes of created dataset."""
        dataset = FashionMNIST(data_dir="test_data/")
        self.assertEqual(len(dataset), 70000)
        self.assertEqual(dataset.x.shape[1], 784)
        self.assertEqual(len(dataset.data_splits), 5)

    def test_UCIGlass(self) -> None:
        """Check attribute shapes of created dataset."""
        dataset = UCIGlass(data_dir="test_data/")
        self.assertEqual(len(dataset), 213)
        self.assertEqual(dataset.x.shape[1], 10)
        self.assertEqual(len(dataset.data_splits), 5)

    def test_UCIParkinsons(self) -> None:
        """Check attribute shapes of created dataset."""
        dataset = UCIParkinsons(data_dir="test_data/")
        self.assertEqual(len(dataset), 195)
        self.assertEqual(dataset.x.shape[1], 22)
        self.assertEqual(len(dataset.data_splits), 5)

    def test_UCISeeds(self) -> None:
        """Check attribute shapes of created dataset."""
        dataset = UCISeeds(data_dir="test_data/")
        self.assertEqual(len(dataset), 209)
        self.assertEqual(dataset.x.shape[1], 7)
        self.assertEqual(len(dataset.data_splits), 5)

    def test_StriatumDataset(self) -> None:
        """Check attribute shapes of created dataset."""
        dataset = StriatumDataset(data_dir="test_data/")
        self.assertEqual(len(dataset), 20000)
        self.assertEqual(dataset.x.shape[1], 272)
        self.assertEqual(len(dataset.data_splits), 5)

    def test_GaussianCloudsDataset(self) -> None:
        """Check attribute shapes of created dataset."""
        dataset = GaussianCloudsDataset(data_dir="test_data/")
        self.assertEqual(len(dataset), 11000)
        self.assertEqual(dataset.x.shape[1], 2)
        self.assertEqual(len(dataset.data_splits), 5)

    def test_Checkerboard2x2Dataset(self) -> None:
        """Check attribute shapes of created dataset."""
        dataset = Checkerboard2x2Dataset(data_dir="test_data/")
        self.assertEqual(len(dataset), 2000)
        self.assertEqual(dataset.x.shape[1], 2)
        self.assertEqual(len(dataset.data_splits), 5)

    def test_Checkerboard4x4Dataset(self) -> None:
        """Check attribute shapes of created dataset."""
        dataset = Checkerboard4x4Dataset(data_dir="test_data/")
        self.assertEqual(len(dataset), 2000)
        self.assertEqual(dataset.x.shape[1], 2)
        self.assertEqual(len(dataset.data_splits), 5)

    def test_CreditCardDataset(self) -> None:
        """Check attribute shapes of created dataset."""
        dataset = CreditCardDataset(data_dir="test_data/")
        self.assertEqual(len(dataset), 284807)
        self.assertEqual(dataset.x.shape[1], 30)
        self.assertEqual(len(dataset.data_splits), 5)
