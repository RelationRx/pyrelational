"""Unit tests for pyrelational.datasets.regression datasets
"""
import pytest
import torch

from pyrelational.data import GenericDataManager
from pyrelational.datasets import (
    SynthReg1,
    SynthReg2,
    DiabetesDataset,
    UCIHousing,
    UCIConcrete,
    UCIEnergy,
    UCIPower,
    UCIWine,
    UCIYacht,
    UCIAirfoil,
)


def test_SynthReg1():
    dataset = SynthReg1()
    assert len(dataset) == 1000
    assert len(dataset.data_splits) == 5

    dataset = SynthReg1(n_splits=3, size=500)
    assert len(dataset) == 500
    assert len(dataset.data_splits) == 3


def test_SynthReg2():
    dataset = SynthReg2()
    assert len(dataset) == 1000
    assert len(dataset.data_splits) == 5

    dataset = SynthReg2(n_splits=3, size=500)
    assert len(dataset) == 500
    assert len(dataset.data_splits) == 3


def test_DiabetesDataset():
    dataset = DiabetesDataset()
    assert len(dataset) == 442
    assert len(dataset.data_splits) == 5


def test_UCIHousing():
    dataset = UCIHousing()
    assert len(dataset) == 505
    assert dataset.x.shape[1] == 13
    assert len(dataset.data_splits) == 5


def test_UCIConcrete():
    dataset = UCIConcrete()
    assert len(dataset) == 1030
    assert dataset.x.shape[1] == 8
    assert len(dataset.data_splits) == 5


def test_UCIEnergy():
    dataset = UCIEnergy()
    assert len(dataset) == 768
    assert dataset.x.shape[1] == 9
    assert len(dataset.data_splits) == 5


def test_UCIPower():
    dataset = UCIPower()
    assert len(dataset) == 9568
    assert dataset.x.shape[1] == 4
    assert len(dataset.data_splits) == 5


def test_UCIWine():
    dataset = UCIWine()
    assert len(dataset) == 1598
    assert dataset.x.shape[1] == 11
    assert len(dataset.data_splits) == 5


def test_UCIYacht():
    dataset = UCIYacht()
    assert len(dataset) == 306
    assert dataset.x.shape[1] == 6
    assert len(dataset.data_splits) == 5


def test_UCIAirfoil():
    dataset = UCIAirfoil()
    assert len(dataset) == 1502
    assert dataset.x.shape[1] == 5
    assert len(dataset.data_splits) == 5
