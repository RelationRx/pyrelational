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

def test_init_and_basic_details():
    gdm = get_classification_dataset(50)
    assert gdm.loader_batch_size == 10
    assert len(gdm.l_indices) == 50
