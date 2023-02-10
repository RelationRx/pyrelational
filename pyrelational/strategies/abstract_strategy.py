"""This module defines the interface for an abstract active learning strategy
which is composed of defining a `__call__` function which
suggests observations to be labelled. In the default case the `__call__`
is the composition of a informativeness function which assigns a measure of
informativeness to unlabelled observations and a selection algorithm which chooses
what observations to present to the oracle
"""
import inspect
import logging
from abc import ABC
from typing import Any, Callable, Dict, List

from pyrelational.data_managers import DataManager
from pyrelational.model_managers import ModelManager

logger = logging.getLogger()


# Trick mypy into not applying contravariance rules to inputs by defining
# __call__ method as a value, rather than a function.  See also
# https://github.com/python/mypy/issues/8795
def _call_unimplemented(self: Any, *input: Any) -> List[int]:
    r"""Defines the computation performed at every call.
    Should be overridden by all subclasses.
    .. note::
        Although the recipe for __call__ needs to be defined within
        this function, one should call the :class:`Strategy` instance afterwards
        instead of this since the former takes care of running the
        registered hooks while the latter silently ignores them.
    """
    raise NotImplementedError(f'Strategy [{type(self).__name__}] is missing the required "__call__" function')


class Strategy(ABC):
    """
    This module defines an abstract active learning strategy.

    Any strategy should be a subclass of this class and override the `__call__` method to suggest observations
    to be labeled. In the general case `__call__` would be the composition of an informativeness function,
    which assigns a measure of informativeness to unlabelled observations, and a selection algorithm which
    chooses what observations to present to the oracle.

    The user defined __call__ method must have a "num_annotate" argument
    """

    def __init__(self, *args: Any, **kwargs: Any):
        super(Strategy, self).__init__()

    __call__: Callable[..., List[int]] = _call_unimplemented

    def suggest(self, num_annotate: int, *args: Any, **kwargs: Any) -> List[int]:
        """
        Filter kwargs and feed arguments to the __call__ method to return unlabelled observations to be labelled
        as a list of dataset indices.

        :param num_annotate: number of samples to annotate
        :param args: any arguments needed by private suggest method
        :param kwargs: any kwargs (filtered to match internal suggest inputs)
        :return: list of indices of samples to query from oracle
        """
        filtered_kwargs = self._filter_kwargs(**kwargs)
        return self(num_annotate=num_annotate, *args, **filtered_kwargs)

    @staticmethod
    def train_and_infer(data_manager: DataManager, model_manager: ModelManager[Any, Any]) -> Any:
        """
        Train the model on the currently labelled subset of the data and produces an output that can be used in
        model uncertainty based strategies.

        :param data_manager: reference to data_manager which will supply data to train model
            and the unlabelled observations
        :param model_manager: Model with generic model interface that will be trained and used to produce
            output of this method
        :return: output of the model
        """
        model_manager.train(data_manager.get_labelled_loader(), data_manager.get_validation_loader())
        output = model_manager(data_manager.get_unlabelled_loader())
        return output

    def _filter_kwargs(self, **kwargs: Any) -> Dict[str, Any]:
        """
        Filter kwargs such that they match the step signature of the concrete strategy.

        :param kwargs: keyword arguments to filter
        :return: filtered keyword arguments
        """
        _params = (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
        _sign_params = inspect.signature(self.__call__).parameters
        filtered_kwargs = {
            k: v for k, v in kwargs.items() if (k in _sign_params.keys() and _sign_params[k].kind not in _params)
        }
        return filtered_kwargs

    def __repr__(self) -> str:
        """Return name of class."""
        return self.__class__.__name__

    def __str__(self) -> str:
        """Pretty prints strategy"""
        str_out = f"Strategy: {self.__repr__()}"
        return str_out
