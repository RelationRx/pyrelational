"""This module defines the interface for a generic active learning strategy
which is composed of defining an `active_learning_step` function which
suggests observations to be labeled. In the default case the `active_learning_step`
is the composition of a informativeness function which assigns a measure of
informativenes to unlabelled observations and a selection algorithm which chooses
what observations to present to the oracle
"""
import logging
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from tabulate import tabulate
from torch.utils.data import DataLoader

from pyrelational.data.data_manager import GenericDataManager
from pyrelational.models.generic_model import GenericModel
from pyrelational.pipeline.pipeline import GenericPipeline

logger = logging.getLogger()


class GenericActiveLearningStrategy(ABC):
    """This module defines the interface for a generic active learning strategy
    which is composed of defining an `active_learning_step` function which
    suggests observations to be labeled. In the default case the `active_learning_step`
    is the composition of a informativeness function which assigns a measure of
    informativenes to unlabelled observations and a selection algorithm which chooses
    what observations to present to the oracle

    :param: pipeline: a reference to a PyRelational pipeline
            used to access pipeline meta data that may be useful for implementing
            complex active learning strategies
    """

    def __init__(self, pipeline: GenericPipeline = None):
        super(GenericActiveLearningStrategy, self).__init__()
        self._pipeline = pipeline

    @abstractmethod
    def active_learning_step(self, num_annotate: int) -> List[int]:
        """Implements a single step of the active learning strategy stopping and returning the
        unlabelled observations to be labelled as a list of dataset indices

        :param num_annotate: number of observations from u to suggest for labelling
        """
        pass

    @property
    def pipeline(self) -> GenericPipeline:
        return self._pipeline

    @pipeline.setter
    def pipeline(self, pipeline: GenericPipeline):
        self._pipeline = pipeline

    def __str__(self) -> str:
        """Pretty prints strategy"""
        str_out = f"Strategy: {self.__name__}"
        return str_out
