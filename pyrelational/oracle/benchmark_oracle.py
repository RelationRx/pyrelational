import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from pyrelational.data.data_manager import DataManager

from .generic_oracle import GenericOracle


class BenchmarkOracle(GenericOracle):
    """An dummy oracle designed for evaluating strategies in R&D settings,
    it assumes that all of the observations are sufficiently annotated and
    returns those annotations when queried.
    """

    def __init__(self):
        super(BenchmarkOracle, self).__init__()

    def query_target_value(self, data_manager: DataManager, idx: int) -> Any:
        """Default method is to simply return the target in the dataset

        :param data_manager: reference to the data_manager which will load the observation if necessary
        :param idx: index to observation which we want to query an annotation

        :return: the output of the oracle (the target value already in the dataset)
        """
        target_value = data_manager.get_sample(idx)[1]
        return target_value
