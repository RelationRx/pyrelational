"""Unit tests for uci dataset downloader
"""

import pytest
import torch

from pyrelational.datasets import UCIDatasets


def test_UCIDatasets():
    dataset = UCIDatasets("glass", data_dir="test_data/", n_splits=10)
    assert dataset.n_splits == 10
    assert len(dataset.data_splits) == 10

    # get split method
    trainsplit = dataset.get_split(train=True)
    assert len(trainsplit) == len(dataset.data_splits[0][0])  # check len train
    testsplit = dataset.get_split(train=False)
    assert len(testsplit) == len(dataset.data_splits[0][1])  # check len test

    # get full split method
    split = dataset.get_full_split()
    full_split_length = len(trainsplit) + len(testsplit)
    assert len(split) == full_split_length

    # get simple dataset method
    torch_dataset = dataset.get_simple_dataset()
    assert len(torch_dataset) == full_split_length
