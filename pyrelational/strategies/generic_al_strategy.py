"""This module defines the interface for a generic active learning strategy
which is composed of defining an `active_learning_step` function which
suggests observations to be labeled. In the default case the `active_learning_step`
is the composition of a informativeness function which assigns a measure of
informativenes to unlabelled observations and a selection algorithm which chooses
what observations to present to the oracle
"""
import inspect
import logging
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from tabulate import tabulate
from torch.utils.data import DataLoader

logger = logging.getLogger()


class GenericActiveLearningStrategy(ABC):
    """This module defines the interface for a generic active learning strategy
    which is composed of defining an `active_learning_step` function which
    suggests observations to be labeled. In the default case the `active_learning_step`
    is the composition of a informativeness function which assigns a measure of
    informativenes to unlabelled observations and a selection algorithm which chooses
    what observations to present to the oracle

    """

    def __init__(self):
        super(GenericActiveLearningStrategy, self).__init__()

    @abstractmethod
    def active_learning_step(self, *args, **kwargs) -> List[int]:
        """Implements a single step of the active learning strategy stopping and returning the
        unlabelled observations to be labelled as a list of dataset indices

        :param num_annotate: number of observations from u to suggest for labelling
        """
        pass

    def _filter_kwargs(self, **kwargs: Any) -> Dict[str, Any]:
        """filter kwargs such that they match the active_learning_step signature of the concrete strategy."""

        # filter all parameters based on update signature except those of
        # type VAR_POSITIONAL (*args) and VAR_KEYWORD (**kwargs)
        _params = (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
        _sign_params = inspect.signature(self.active_learning_step).parameters
        filtered_kwargs = {
            k: v for k, v in kwargs.items() if (k in _sign_params.keys() and _sign_params[k].kind not in _params)
        }

        exists_var_keyword = any(v.kind == inspect.Parameter.VAR_KEYWORD for v in _sign_params.values())
        # if no kwargs filtered, return all kwargs as default
        if not filtered_kwargs and not exists_var_keyword:
            # no kwargs in update signature -> don't return any kwargs
            filtered_kwargs = {}
        elif exists_var_keyword:
            # kwargs found in update signature -> return all kwargs to be sure to not omit any.
            # filtering logic is likely implemented within the update call.
            filtered_kwargs = kwargs
        return filtered_kwargs

    def __repr__(self) -> str:
        """Pretty prints strategy"""
        str_out = f"{self.__name__}"
        return str_out

    def __str__(self) -> str:
        """Pretty prints strategy"""
        str_out = f"Strategy: {self.__name__}"
        return str_out
