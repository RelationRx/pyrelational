"""Unit tests for data manager
"""
from typing import Tuple

import pytest
import torch

from pyrelational.data import GenericDataManager
from tests.test_utils import (
    DiabetesDataset,
    get_classification_dataset,
    get_regression_dataset,
)


def test_init_and_basic_details():
    gdm = get_classification_dataset(50)
    assert gdm.loader_batch_size == 10
    assert len(gdm.l_indices) == 50


def test_print():
    gdm = get_classification_dataset(50)
    assert repr(gdm) == "GenericDataManager"

    out = "GenericDataManager\nTraining set size: 400\nLabelled: 50, Unlabelled: 350\nPercentage Labelled: 0.125"
    assert str(gdm) == out


def test_get_train_set():
    gdm = get_classification_dataset(50)
    tl = gdm.get_train_set()
    assert len(tl) == 400


def test_update_train_labels():
    gdm = get_classification_dataset(50)

    random_u_sindex = gdm.u_indices[0]
    len_gdm_l = len(gdm.l_indices)
    len_gdm_u = len(gdm.u_indices)

    gdm.update_train_labels([random_u_sindex])
    assert random_u_sindex in gdm.l_indices
    assert len(gdm.l_indices) > len_gdm_l
    assert len(gdm.u_indices) < len_gdm_u


def test_percentage_labelled():
    gdm = get_classification_dataset()
    percentage = gdm.percentage_labelled()
    assert percentage == pytest.approx(0.1, 0.05)


def test_get_dataset_size():
    gdm = get_classification_dataset()
    ds_size = len(gdm)
    assert ds_size == 569


def test_train_and_test_indices_both_supplied():
    """Test if train and test indices are both supplied, that remaining indices are unused."""

    valid_dm = get_regression_dataset(use_train=True, use_validation=True, use_test=True)
    no_valid_dm = get_regression_dataset(use_train=True, use_validation=False, use_test=True)
    assert len(valid_dm.train_indices) == 350
    assert len(valid_dm.validation_indices) == 50
    assert len(valid_dm.test_indices) == 42
    assert len(no_valid_dm.train_indices) == 350
    assert no_valid_dm.validation_indices is None
    assert len(no_valid_dm.test_indices) == 42


def test_no_train_indices():
    """Test if no train indices are supplied (but test indices are), remaining indices become train set."""
    ds = DiabetesDataset()
    valid_dm = get_regression_dataset(use_train=False, use_validation=True, use_test=True)
    no_valid_dm = get_regression_dataset(use_train=False, use_validation=False, use_test=True)
    assert len(valid_dm.train_indices) == len(ds) - 42 - 50
    assert len(no_valid_dm.train_indices) == len(ds) - 42


def test_no_test_indices():
    """Test if no test indices are supplied (but train indices are), remaining indices become test set."""
    ds = DiabetesDataset()
    valid_dm = get_regression_dataset(use_train=True, use_validation=True, use_test=False)
    no_valid_dm = get_regression_dataset(use_train=True, use_validation=False, use_test=False)
    assert len(valid_dm.test_indices) == len(ds) - 350 - 50
    assert len(no_valid_dm.test_indices) == len(ds) - 350


def test_no_test_and_no_train_indices():
    """Test if no test and no train indices are supplied, error is raised."""
    ds = DiabetesDataset()
    _, valid_ds, _ = torch.utils.data.random_split(ds, [350, 50, 42])
    valid_indices = valid_ds.indices
    with pytest.raises(ValueError) as case5:
        GenericDataManager(
            DiabetesDataset(),
            validation_indices=valid_indices,
            loader_batch_size=10,
        )
    assert str(case5.value) == "No train or test specified, too ambigious to set values"
    with pytest.raises(ValueError) as case5:
        GenericDataManager(
            DiabetesDataset(),
            loader_batch_size=10,
        )
    assert str(case5.value) == "No train or test specified, too ambigious to set values"


def test_empty_test_set():
    """Test error occurs when we produce a datamanager with an empty test set."""
    ds = DiabetesDataset()
    train_ds, valid_ds, _ = torch.utils.data.random_split(ds, [400, 42, 0])
    valid_indices = valid_ds.indices
    train_indices = train_ds.indices
    with pytest.raises(ValueError) as e_info:
        GenericDataManager(
            ds,
            validation_indices=valid_indices,
            train_indices=train_indices,
            loader_batch_size=10,
        )
    assert str(e_info.value) == "The test set is empty"


def test_empty_train_set():
    """
    Testing that we throw an error when we produce a datamanager with
    an empty train set
    """

    # Case 1: user gives empty train set
    ds = DiabetesDataset()

    # Artificially create a test leakage as example
    train_ds, valid_ds, test_ds = torch.utils.data.random_split(ds, [0, 400, 42])
    train_indices = train_ds.indices
    valid_indices = valid_ds.indices
    test_indices = test_ds.indices

    with pytest.raises(ValueError) as e_info:
        GenericDataManager(
            ds,
            train_indices=train_indices,
            validation_indices=valid_indices,
            test_indices=test_indices,
            loader_batch_size=10,
        )
    assert str(e_info.value) == "The train set is empty"

    # Case 2: empty train set produced by resolving of indices
    _, valid_ds, test_ds = torch.utils.data.random_split(ds, [0, 400, 42])
    valid_indices = valid_ds.indices
    test_indices = test_ds.indices

    with pytest.raises(ValueError) as e_info:
        GenericDataManager(
            ds,
            validation_indices=valid_indices,
            test_indices=test_indices,
            loader_batch_size=10,
        )
    assert str(e_info.value) == "The train set is empty"


def test_resolving_dataset_check_split_leaks():
    """Check we throw an error for data split leaks"""
    ds = DiabetesDataset()

    # Artificially create a test leakage as example
    train_ds, valid_ds, test_ds = torch.utils.data.random_split(ds, [350, 50, 42])
    train_indices = train_ds.indices
    valid_indices = valid_ds.indices
    test_indices = valid_ds.indices  # we have leak here

    # Check no leaks
    with pytest.raises(ValueError) as e_info:
        GenericDataManager(
            ds,
            train_indices=train_indices,
            validation_indices=valid_indices,
            test_indices=test_indices,
            loader_batch_size=10,
        )
    assert str(e_info.value) == "There is an overlap between the split indices supplied"
