"""Unit tests for data manager
"""
import copy
from unittest import TestCase

import pytest
import torch
from parameterized import parameterized

from pyrelational.data_managers import DataManager
from tests.test_utils import (
    DiabetesDataset,
    get_classification_dataset,
    get_regression_dataset,
)


class TestDataManager(TestCase):
    """Class containing unit test on pyrelational data manager."""

    def test_init_and_attributes_size(self) -> None:
        """Check initialisation runs and data manager attributes have correct size."""
        gdm = get_classification_dataset(50)
        self.assertEqual(gdm.loader_batch_size, 10)
        self.assertEqual(len(gdm.l_indices), 50)

    def test_repr_and_print(self) -> None:
        """Check __repr__ and pretty print are correct."""
        gdm = get_classification_dataset(50)
        self.assertEqual(repr(gdm), "DataManager")

        out = "DataManager\nTraining set size: 400\nLabelled: 50, Unlabelled: 350\nPercentage Labelled: 12.500"
        self.assertEqual(str(gdm), out)

    def test_get_split_sets(self) -> None:
        """Check train/val/test set getter output has expected size."""
        gdm = get_classification_dataset(
            50,
            train_val_test_split=(400, 100, 69),
        )
        self.assertEqual(len(gdm.get_train_set()), 400)
        self.assertEqual(len(gdm.get_validation_set()), 100)
        self.assertEqual(len(gdm.get_test_set()), 69)

    def test_update_train_labels(self) -> None:
        """Check update of labelled/unlabelled sets has intended effect."""
        gdm = get_classification_dataset(50)

        random_u_sindex = gdm.u_indices[0]
        len_gdm_l = len(gdm.l_indices)
        len_gdm_u = len(gdm.u_indices)
        gdm.update_train_labels([random_u_sindex])

        self.assertIn(random_u_sindex, gdm.l_indices)
        self.assertNotIn(random_u_sindex, gdm.u_indices)
        self.assertEqual(len(gdm.l_indices), len_gdm_l + 1)
        self.assertEqual(len(gdm.u_indices), len_gdm_u - 1)

    def test_update_labels(self) -> None:
        """Check that dataset object is properly updated by set_target_value method."""
        gdm = get_regression_dataset()
        before = copy.deepcopy(gdm[0][-1])
        gdm.set_target_value(0, -32)
        self.assertNotEqual(before, gdm[0][-1])
        self.assertEqual(-32, gdm[0][-1])

    def test_percentage_labelled(self) -> None:
        """Check default labelled percentage."""
        gdm = get_classification_dataset()
        percentage = gdm.get_percentage_labelled()
        self.assertEqual(percentage, pytest.approx(10, 5))

    def test_get_dataset_size(self) -> None:
        """Check dataset size returned is correct."""
        gdm = get_classification_dataset()
        self.assertEqual(len(gdm), 569)

    @parameterized.expand(
        [
            (True,),
            (False,),
        ]
    )
    def test_train_and_test_indices_both_supplied(self, use_validation: bool) -> None:
        """Test that if train and test indices are both supplied, then remaining indices are unused."""
        dm = get_regression_dataset(
            use_train=True,
            use_validation=use_validation,
            use_test=True,
            train_val_test_split=(350, 50, 42),
        )
        self.assertEqual(len(dm.train_indices), 350)
        self.assertEqual(len(dm.test_indices), 42)
        if use_validation:
            self.assertEqual(len(dm.validation_indices), 50)
        else:
            self.assertIsNone(dm.validation_indices)

    @parameterized.expand(
        [
            (True,),
            (False,),
        ]
    )
    def test_no_train_indices(self, use_validation: bool) -> None:
        """Test if no train indices are supplied (but test indices are), remaining indices become train set."""
        ds = DiabetesDataset()
        dm = get_regression_dataset(
            use_train=False,
            use_validation=use_validation,
            use_test=True,
            train_val_test_split=(350, 50, 42),
        )
        expected_len = (len(ds) - 92) if use_validation else (len(ds) - 42)
        self.assertEqual(len(dm.train_indices), expected_len)

    @parameterized.expand(
        [
            (True,),
            (False,),
        ]
    )
    def test_no_test_indices(self, use_validation: bool) -> None:
        """Test if no test indices are supplied (but train indices are), remaining indices become test set."""
        ds = DiabetesDataset()
        dm = get_regression_dataset(
            use_train=True,
            use_validation=use_validation,
            use_test=False,
            train_val_test_split=(350, 50, 42),
        )
        expected_len = (len(ds) - 400) if use_validation else (len(ds) - 350)
        self.assertEqual(len(dm.test_indices), expected_len)

    @parameterized.expand(
        [
            (True,),
            (False,),
        ]
    )
    def test_no_test_and_no_train_indices(self, use_validation: bool) -> None:
        """Test if no test and no train indices are supplied, error is raised."""
        valid_indices = list(range(50)) if use_validation else None
        with pytest.raises(ValueError) as case:
            DataManager(
                DiabetesDataset(),
                validation_indices=valid_indices,
                loader_batch_size=10,
            )
        self.assertEqual(str(case.value), "No train or test specified, too ambiguous to set values")

    def test_empty_test_set(self) -> None:
        """Test error occurs when we produce a datamanager with an empty test set."""
        ds = DiabetesDataset()
        train_ds, valid_ds, _ = torch.utils.data.random_split(ds, [400, 42, 0])
        valid_indices = valid_ds.indices
        train_indices = train_ds.indices
        with pytest.raises(ValueError) as e_info:
            DataManager(
                ds,
                validation_indices=valid_indices,
                train_indices=train_indices,
                loader_batch_size=10,
            )
        self.assertEqual(str(e_info.value), "The test set is empty")

    @parameterized.expand(
        [
            (True,),
            (False,),
        ]
    )
    def test_fail_on_empty_train_set(self, feed_empty_train_set: bool) -> None:
        """
        Test that an error is thrown when one tries to instantiate a data manager with an empty train set.

        This should fail is two cases:
            1- user gives empty train set
            2- empty train set produced by resolving of indices
        """
        ds = DiabetesDataset()
        train_ds, valid_ds, test_ds = torch.utils.data.random_split(ds, [0, 400, 42])
        with pytest.raises(ValueError) as e_info:
            DataManager(
                ds,
                train_indices=train_ds.indices if feed_empty_train_set else None,
                validation_indices=valid_ds.indices,
                test_indices=test_ds.indices,
                loader_batch_size=10,
            )
        self.assertEqual(str(e_info.value), "The train set is empty")

    def test_resolving_dataset_check_split_leaks(self) -> None:
        """Check we throw an error for data split leaks."""
        ds = DiabetesDataset()
        # Artificially create a test leakage as example
        train_ds, valid_ds, test_ds = torch.utils.data.random_split(ds, [350, 50, 42])
        # Check no leaks
        with pytest.raises(ValueError) as e_info:
            DataManager(
                ds,
                train_indices=train_ds.indices,
                validation_indices=valid_ds.indices,
                test_indices=valid_ds.indices,  # we have leak here
                loader_batch_size=10,
            )
        self.assertEqual(str(e_info.value), "There is an overlap between the split indices supplied")
