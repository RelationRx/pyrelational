"""
This file contains the implementation of an abstract oracle interface for PyRelationAL
"""
from abc import ABC, abstractmethod
from typing import Any, List

from pyrelational.data_managers.data_manager import DataManager


class Oracle(ABC):
    """
    An abstract class acting as an interface for implementing concrete oracles
    that can interact with a pyrelational pipeline
    """

    def __init__(self) -> None:
        super(Oracle, self).__init__()

    @staticmethod
    def update_target_value(data_manager: DataManager, idx: int, value: Any) -> None:
        """Update the target value for the observation denoted by the index

        :param data_manager: reference to the data_manager whose dataset we want to update
        :param idx: index to the observation we want to update
        :param value: value to update the observation with
        """
        data_manager.set_target_value(idx=idx, value=value)

    @staticmethod
    def update_target_values(data_manager: DataManager, indices: List[int], values: List[Any]) -> None:
        """Updates the target values of the observations at the supplied indices

        :param data_manager: reference to the data_manager whose dataset we want to update
        :param indices: list of indices to observations whose target values we want to update
        :param values: list of values which we want to assign to the corresponding observations in indices
        """
        for idx, val in zip(indices, values):
            data_manager.set_target_value(idx=idx, value=val)

    @staticmethod
    def update_annotations(data_manager: DataManager, indices: List[int]) -> None:
        """Calls upon the data_manager to update the set of labelled indices with those supplied
        as arguments. It will move the observations associated with the supplied indices from the
        unlabelled set to the labelled set. By default, any indices supplied that are already in
        the labelled set are untouched.

        Note this does not change the target values of the indices, this is handled by a method
        in the oracle.

        :param data_manager: reference to the data_manager whose sets we are adjusting
        :param indices: list of indices selected for labelling
        """
        data_manager.update_train_labels(indices)

    @abstractmethod
    def query_target_value(self, data_manager: DataManager, idx: int) -> Any:
        """Method that needs to be overridden to obtain the annotations for the input index

        :param data_manager: reference to the data_manager which will load the observation if necessary
        :param idx: index to observation which we want to query an annotation

        :return: the output of the oracles
        """
        pass

    def update_dataset(self, data_manager: DataManager, indices: List[int]) -> None:
        """
        This method serves to obtain labels for the supplied indices and update the
        target values in the corresponding observations of the data manager

        :param data_manager: reference to DataManager whose dataset we intend to update
        :param indices: list of indices to observations we want updated
        """
        for idx in indices:
            target_val = self.query_target_value(data_manager=data_manager, idx=idx)
            self.update_target_value(data_manager=data_manager, idx=idx, value=target_val)
        self.update_annotations(data_manager=data_manager, indices=indices)

    def __str__(self) -> str:
        """Return class name."""
        return self.__class__.__name__
