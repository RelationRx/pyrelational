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

from pyrelational.data import DataManager
from pyrelational.models import ModelManager

logger = logging.getLogger()


class Strategy(ABC):
    """This module defines the interface for a generic active learning strategy
    which is composed of defining an `active_learning_step` function which
    suggests observations to be labeled. In the default case the `active_learning_step`
    is the composition of a informativeness function which assigns a measure of
    informativenes to unlabelled observations and a selection algorithm which chooses
    what observations to present to the oracle

    """

    def __init__(self):
        super(Strategy, self).__init__()

    @abstractmethod
    def active_learning_step(self, *args, **kwargs) -> List[int]:
        """Implements a single step of the active learning strategy stopping and returning the
        unlabelled observations to be labelled as a list of dataset indices

        :param num_annotate: number of observations from u to suggest for labelling
        """
        pass

    def train_and_infer(self, data_manager: DataManager, model: ModelManager) -> Any:
        """Trains the model on the currently labelled subset of the data and produces
        an output that can be used in model uncertainty based strategies

        :param data_manager: reference to data_manager which will supply data to train model
            and the unlabelled observations
        :param model: Model with generic model interface that will be trained and used to produce
            output of this method
        """
        model.train(data_manager.get_labelled_loader(), data_manager.get_validation_loader())
        output = model(data_manager.get_unlabelled_loader())
        return output

    def _filter_kwargs(self, **kwargs: Any) -> Dict[str, Any]:
        """filter kwargs such that they match the active_learning_step signature of the concrete strategy."""

        # filter all parameters based on update signature except those of
        # type VAR_POSITIONAL (*args) and VAR_KEYWORD (**kwargs)
        _params = (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
        _sign_params = inspect.signature(self.active_learning_step).parameters
        filtered_kwargs = {
            k: v for k, v in kwargs.items() if (k in _sign_params.keys() and _sign_params[k].kind not in _params)
        }
        return filtered_kwargs

    def __repr__(self):
        return self.__class__.__name__

    def __str__(self) -> str:
        """Pretty prints strategy"""
        str_out = f"Strategy: {self.__repr__}"
        return str_out
