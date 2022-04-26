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