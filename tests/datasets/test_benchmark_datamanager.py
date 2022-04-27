"""Unit tests for benchmark datamanager functions
"""

import pytest

from pyrelational.datasets import BreastCancerDataset, DiabetesDataset
from pyrelational.datasets.benchmark_datamanager import (
    create_classification_cold_start,
    create_regression_cold_start,
    create_warm_start,
)


def test_create_warm_start_classification():
    dataset = BreastCancerDataset()
    train_indices = list(dataset.data_splits[0][0])
    test_indices = list(dataset.data_splits[0][1])
    dm = create_warm_start(dataset, train_indices=train_indices, test_indices=test_indices)
    assert len(dm) == 569


def test_create_warm_start_regression():
    dataset = DiabetesDataset()
    train_indices = list(dataset.data_splits[0][0])
    test_indices = list(dataset.data_splits[0][1])
    dm = create_warm_start(dataset, train_indices=train_indices, test_indices=test_indices)
    assert len(dm) == 442


def test_create_classification_cold_start():
    dataset = BreastCancerDataset()
    train_indices = list(dataset.data_splits[0][0])
    test_indices = list(dataset.data_splits[0][1])
    dm = create_classification_cold_start(dataset, train_indices=train_indices, test_indices=test_indices)
    assert len(dm) == 569
    assert len(dm.l_indices) == 2


def test_create_regression_cold_start():
    dataset = DiabetesDataset()
    train_indices = list(dataset.data_splits[0][0])
    test_indices = list(dataset.data_splits[0][1])
    dm = create_regression_cold_start(dataset, train_indices=train_indices, test_indices=test_indices)
    assert len(dm) == 442
    assert len(dm.l_indices) == 2
