"""This module defines the acquisition manager which utilises
the data manager, sampling functions, and model to create acquisition
functions and general arbiters of the active learning pipeline
"""
import logging
from abc import ABC
from collections import defaultdict
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from tabulate import tabulate
from torch.utils.data import DataLoader

from pyrelational.data_managers.data_manager import DataManager
from pyrelational.model_managers.abstract_model_manager import ModelManager
from pyrelational.oracles.abstract_oracle import Oracle
from pyrelational.oracles.benchmark_oracle import BenchmarkOracle
from pyrelational.strategies.abstract_strategy import Strategy

logger = logging.getLogger()


class Pipeline(ABC):
    """
    The pipeline facilitates the communication between

        - DataManager
        - ModelManager,
        - Strategy,
        - Oracle (Optional)

    To enact a generic active learning cycle.
    """

    def __init__(
        self,
        data_manager: DataManager,
        model_manager: ModelManager[Any, Any],
        strategy: Strategy,
        oracle: Optional[Oracle] = None,
    ):
        """
        :param data_manager: A pyrelational data manager
            which keeps track of what has been labelled and creates data loaders for
            active learning
        :param model_manager: A pyrelational model manager which handles the instantiation, training, testing of
            a machine learning model for the data in the data manager
        :param strategy: A pyrelational active learning strategy
            implements the informativeness measure and the selection algorithm being used
        :param oracle: An oracle instance
            interfaces with various concrete oracle to obtain labels for observations
            suggested by the strategy
        """
        super(Pipeline, self).__init__()
        self.data_manager = data_manager
        self.model_manager = model_manager
        self.strategy = strategy
        self.oracle: Oracle = BenchmarkOracle() if oracle is None else oracle

        # Pipeline meta properties
        self.iteration = 0

        # Data structures for logging values of interest
        self.performances: Dict[Union[int, str], Dict[str, float]] = defaultdict(dict)
        self.labelled_by: Dict[int, Dict[str, Union[str, int]]] = defaultdict(dict)
        self.log_labelled_by(data_manager.l_indices, tag="initialisation")

    def compute_theoretical_performance(self, test_loader: Optional[DataLoader[Any]] = None) -> Dict[str, float]:
        """
        Returns the performance of the full labelled dataset against the
        test data. Typically used for evaluation to establish theoretical benchmark
        of model performance given all available training data is labelled. The
        "horizontal" line in area under learning curve plots for active learning

        Would not make much sense when we are doing active learning for the real
        situation, hence not part of __init__

        :param test_loader: Pytorch Data Loader with
            test data compatible with model, optional as often the test loader can be
            generated from data_manager but is here for case when it hasn't been defined
            or there is a new test set.

        :return: dictionary containing metric results on test set
        """
        self.model_manager.train(self.train_loader, self.valid_loader)

        # use test loader in data_manager if there is one
        result = self.model_manager.test(self.test_loader if test_loader is None else test_loader)
        result = self.compute_hit_ratio(result)
        self.performances["full"] = result

        # make sure that theoretical best model is not stored
        self.model_manager.reset()
        return self.performances["full"]

    def compute_current_performance(
        self, test_loader: Optional[DataLoader[Any]] = None, query: Optional[List[int]] = None
    ) -> None:
        """
        Compute performance of model.

        :param test_loader: Pytorch Data Loader with
            test data compatible with model, optional as often the test loader can be
            generated from data_manager but is here for case when it hasn't been defined
            or there is a new test set.
        :param query: List of indices selected for labelling. Used for calculating hit ratio metric
        :return: dictionary containing metric results on test set
        """
        if not self.model_manager.is_trained():  # no AL steps taken so far
            self.model_manager.train(self.l_loader, self.valid_loader)

        # use test loader in data_manager if there is one
        result = self.model_manager.test(self.test_loader if test_loader is None else test_loader)
        result = self.compute_hit_ratio(result, query)
        self.performances[self.iteration] = result
        return None

    def compute_hit_ratio(self, result: Dict[str, float], query: Optional[List[int]] = None) -> Dict[str, float]:
        """
        Utility function for computing the hit ratio as used within the current performance
        and theoretical performance methods.

        :param result: Dict or Dict-like of metrics
        :param query: List of indices selected for labelling. Used for calculating hit ratio metric

        :return: updated result dictionary with "hit_ratio" key, corresponding to hit ratio result
        """
        if self.data_manager.top_unlabelled is not None:
            result["hit_ratio"] = (
                np.nan
                if query is None
                else len(set(query) & self.data_manager.top_unlabelled) / len(self.data_manager.top_unlabelled)
            )
        return result

    def step(self, num_annotate: int, *args: Any, **kwargs: Any) -> List[int]:
        """
        Ask the strategy to provide indices of unobserved observations for labelling by the oracle

        :param num_annotate: Number of points to annotate

        :return: list of indexes to label from dataset
        """
        default_kwargs = self.__dict__
        kwargs = {**default_kwargs, **kwargs}  # update kwargs with any user defined ones
        observations_for_labelling = self.strategy.suggest(num_annotate, *args, **kwargs)
        return observations_for_labelling

    def query(self, indices: List[int]) -> None:
        """
        Updates labels based on indices selected for labelling

        :param indices: List of indices selected for labelling
        """
        self.oracle.update_dataset(data_manager=self.data_manager, indices=indices)

        # Logging
        self.iteration += 1
        logger.info("Length of labelled %s" % (len(self.l_indices)))
        logger.info("Length of unlabelled %s" % (len(self.u_indices)))
        logger.info("Percentage labelled %s" % self.percentage_labelled)
        self.log_labelled_by(indices)

    def run(
        self,
        num_annotate: int,
        num_iterations: Optional[int] = None,
        test_loader: Optional[DataLoader[Any]] = None,
        *strategy_args: Any,
        **strategy_kwargs: Any,
    ) -> None:
        """
        Given the number of samples to annotate and a test loader this method will go through the entire
        active learning process of training the model on the labelled set, and recording the current performance
        based on this. Then it will proceed to compute uncertainties for the unlabelled observations, rank them,
        and get the top num_annotate observations labelled to be added to the next iteration's labelled dataset L'.
        This process repeats until there are no observations left in the unlabelled set.

        :param num_annotate: number of observations to get annotated per iteration
        :param num_iterations: number of active learning loop to perform
        :param test_loader: test data with which we evaluate the current state of the model given the labelled set L
        :param strategy_args: optional additional args for strategy call
        :param strategy_kwargs: optional additional kwargs for strategy call
        """
        iter_count = 0
        while len(self.u_indices) > 0:
            iter_count += 1

            # Obtain samples for labelling and pass to the oracle interface if supplied
            observations_for_labelling = self.step(num_annotate, *strategy_args, **strategy_kwargs)

            # Record the current performance
            self.compute_current_performance(
                test_loader=test_loader,
                query=observations_for_labelling,
            )
            self.query(
                observations_for_labelling,
            )
            if (num_iterations is not None) and iter_count == num_iterations:
                break

        # Final update the model and check final test performance
        self.model_manager.train(self.l_loader, self.valid_loader)
        self.compute_current_performance(test_loader=test_loader)

    def summary(self) -> pd.DataFrame:
        """Construct a pandas table of performances of the model over the active learning iterations."""
        keys = sorted(set(self.performances.keys()) - {"full"})
        logger.info("KEYS: {}".format(keys))
        df = []
        if "full" in self.performances:
            columns = ["Iteration"] + sorted(list(self.performances["full"].keys()))
            logger.info("COLUMNS: {}".format(columns))
            logger.info("Full inside")
        else:
            if self.iteration in self.performances:
                columns = ["Iteration"] + sorted(list(self.performances[self.iteration].keys()))
            else:
                columns = ["Iteration"]
            logger.info("COLUMNS: {}".format(columns))
            logger.info("Full Missing")
        for k in keys:
            row: List[Union[str, float]] = [k]
            logger.info(self.performances[k])
            for c in columns[1:]:
                row.append(self.performances[k][c])
            df.append(row)

        pd_df = pd.DataFrame(df, columns=columns)
        return pd_df

    def log_labelled_by(self, indices: List[int], tag: Optional[str] = None) -> None:
        """
        Update the dictionary that records what the observation
        was labelled by. Default behaviour is to map observation to
        iteration at which it was labelled

        :param indices: list of indices selected for labelling
        :param tag: string which indicates what the observations where labelled by
        """
        for indx in indices:
            self.labelled_by[indx]["iteration"] = self.iteration
            self.labelled_by[indx]["source"] = str(self.oracle) if tag is None else tag

    @property
    def u_indices(self) -> List[int]:
        """Indices of unlabelled samples."""
        return self.data_manager.u_indices

    @property
    def u_loader(self) -> DataLoader[Any]:
        """Dataloader containing unlabelled data."""
        return self.data_manager.get_unlabelled_loader()

    @property
    def l_indices(self) -> List[int]:
        """Indices of labelled samples."""
        return self.data_manager.l_indices

    @property
    def l_loader(self) -> DataLoader[Any]:
        """Dataloader containing labelled data."""
        return self.data_manager.get_labelled_loader()

    @property
    def train_loader(self) -> DataLoader[Any]:
        """Dataloader containing train data."""
        return self.data_manager.get_train_loader(full=True)

    @property
    def valid_loader(self) -> Optional[DataLoader[Any]]:
        """Dataloader containing validation data."""
        return self.data_manager.get_validation_loader()

    @property
    def test_loader(self) -> DataLoader[Any]:
        """Dataloader containing test data."""
        return self.data_manager.get_test_loader()

    @property
    def percentage_labelled(self) -> float:
        """Percentage of total available dataset labelled."""
        return self.data_manager.get_percentage_labelled()

    @property
    def dataset_size(self) -> int:
        """Number of total data points."""
        return len(self.data_manager)

    def __repr__(self) -> str:
        return self.__class__.__name__

    def __str__(self) -> str:
        """Pretty prints contents inside ActiveLearningManager based on available attributes"""
        str_dm = str(self.data_manager)
        str_model = str(self.model_manager)
        str_dataset_size = str(self.dataset_size)
        str_percentage_labelled = "%.3f" % self.percentage_labelled

        str_out = self.__repr__()
        str_out += "DataManager: %s \n" % str_dm
        str_out += "Model: %s \n" % str_model
        str_out += "Size of Dataset: %s \n" % str_dataset_size
        str_out += "Percentage of Dataset Labelled for Model: %s \n" % str_percentage_labelled
        if "full" in self.performances:
            str_out += "Theoretical performance: %s \n" % str(self.performances["full"])
        str_out += "Performance history \n"
        str_out += tabulate(self.summary(), tablefmt="pipe", headers="keys")
        return str_out
