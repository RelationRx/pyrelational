"""
Utility to create datamanagers corresponding to different 

"""
import random
import torch
import numpy as np
from sklearn.metrics import pairwise_distances
from classification_datasets import SynthClass1, SynthClass2, SynthClass3
from regression_datasets import SynthReg1, SynthReg2
from pyrelational.data.data_manager import GenericDataManager
from collections import defaultdict


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
    