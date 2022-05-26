"""Utility to create datamanagers corresponding to different AL tasks
"""
import random
from collections import defaultdict

import numpy as np
import torch
from sklearn.metrics import pairwise_distances

from pyrelational.data.data_manager import GenericDataManager


def pick_one_sample_per_class(dataset, train_indices):
    """Utility function to randomly pick one sample per class in the
    training subset of dataset and return their index in the dataset.
    This is used for defining an initial state of the labelled subset
    in the active learning task

    :param train_indices: list or iterable with the indices corresponding
        to the training samples in the dataset
    """
    class2idx = defaultdict(list)
    for idx in train_indices:
        idx_class = int(dataset[idx][1])
        class2idx[idx_class].append(idx)

    class_reps = []
    for idx_class in class2idx.keys():
        random_class_idx = random.choice(class2idx[idx_class])
        class_reps.append(random_class_idx)

    return class_reps


def create_warm_start(dataset, **dm_args):
    """Returns a datamanager with 10% randomly labelled data
    from the train indices. The rest of the observations in the training
    set comprise the unlabelled set of observations. We call this
    initialisation a 'warm start' AL task inspired by
    Konyushkova et al. (2017)

    This can be used both for classification and regression type datasets.

    From Ksenia Konyushkova, Raphael Sznitman, Pascal Fua 'Learning Active
    Learning from Data', NIPS 2017

    :param dataset: A pytorch dataset in the style described
        pyrelational.datasets
    :param dm_args: kwargs for any additional keyword arguments to be passed
        into the initialisation of the datamanager.
    """
    dm = GenericDataManager(dataset, **dm_args)
    return dm


def create_classification_cold_start(dataset, train_indices, test_indices, **dm_args):
    """Returns an AL task for benchmarking classification datasets. The
    AL task will sample an example from each of the classes in the training
    subset of the data.

    Please note the current iteration does not utilise a validation set
    as described in the paper

    :param dataset: A pytorch dataset in the style described
        pyrelational.datasets
    :param train_indices: [int] indices corresponding to observations of dataset
        used for training set
    :param test_indices: [int] indices corresponding to observations of dataset
        used for holdout test set
    :param dm_args: kwargs for any additional keyword arguments to be passed
        into the initialisation of the datamanager.
    """
    labelled_indices = pick_one_sample_per_class(dataset, train_indices)
    dm = GenericDataManager(
        dataset, train_indices=train_indices, test_indices=test_indices, labelled_indices=labelled_indices, **dm_args
    )
    return dm


def create_regression_cold_start(dataset, train_indices, test_indices, **dm_args):
    """Create data manager with 2 labelled data samples, where the data samples
    labelled are the pair that have the largest distance between them

    Please note the current iteration does not utilise a validation set
    as described in the paper

    :param dataset: A pytorch dataset in the style described
        pyrelational.datasets
    :param train_indices: [int] indices corresponding to observations of dataset
        used for training set
    :param test_indices: [int] indices corresponding to observations of dataset
        used for holdout test set
    :param dm_args: kwargs for any additional keyword arguments to be passed
        into the initialisation of the datamanager.
    """
    # Find the two samples within the training subset that have the largest distance between them.
    pair_dists = pairwise_distances(dataset[train_indices][:][0])
    sample1_idx, sample2_idx = np.unravel_index(np.argmax(pair_dists, axis=None), pair_dists.shape)
    sample1_idx = train_indices[sample1_idx]  # map to dataset index from local index
    sample2_idx = train_indices[sample2_idx]

    labelled_indices = [sample1_idx, sample2_idx]
    dm = GenericDataManager(
        dataset,
        train_indices=train_indices,
        test_indices=test_indices,
        labelled_indices=labelled_indices,
        **dm_args,
    )
    return dm
