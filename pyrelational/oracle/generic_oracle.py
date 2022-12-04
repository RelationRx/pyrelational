"""
This file contains the implementation of a generic oracle interface for PyRelationAL
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from pyrelational.data.data_manager import GenericDataManager


class GenericOracle(ABC):
    """An abstract class acting as an interface for implementing concrete oracles
    that can interact with a pyrelational pipeline"""

    def __init__(self):
        super(GenericOracle, self).__init__()

    @abstractmethod
    def update_dataset(data_manager: GenericDataManager, indices: List[int]) -> List[Any]:
        """
        This method serves to obtain labels for the supplied indices and update the
        target values in the corresponding observations of the data manager
        """
        pass
