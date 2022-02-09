"""This module defines the acquisition manager which utilises
the data manager, sampling functions, and model to create acquisition
functions and general arbiters of the active learning pipeline
"""
import logging
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Dict, List, Optional

import pandas as pd
from tabulate import tabulate
from torch.utils.data import DataLoader

from pyrelational.data.data_manager import GenericDataManager
from pyrelational.models.generic_model import GenericModel

logger = logging.getLogger()


class GenericActiveLearningStrategy(ABC):
    """The Acquisition Manager uses the sampling functions and data manager
    to define specific acquisition/active learning strategies and hence
    drive the active learning process

    Args:
        data_manager (pyrelational.data_manager.GenericDataManager): an pyrelational data manager
            which keeps track of what has been labelled and creates data loaders for
            active learning
        model (pyrelational.models.generic_model.GenericModel): An pyrelational model
            which serves as the machine learning model for the data in the
            data manager
    """

    def __init__(self, data_manager: GenericDataManager, model: GenericModel):
        super(GenericActiveLearningStrategy, self).__init__()
        self.data_manager = data_manager
        self.model = model
        self.iteration = 0

        # Datastructures for logging values of interest
        self.performances = defaultdict(dict)
        self.labelled_by = {}
        self.log_labelled_by(data_manager.l_indices, tag="Initialisation")

    def theoretical_performance(self, test_loader: Optional[DataLoader] = None) -> Dict:
        """Returns the performance of the full labelled dataset against the
        test data. Typically used for evaluation to establish theoretical benchmark
        of model performance given all available training data is labelled. The
        "horizontal" line in area under learning curve plots for active learning

        Would not make much sense when we are doing active learning for the real
        situation, hence not part of __init__

        Args:
            test_loader (Optional: torch.utils.data.DataLoader): Pytorch Data Loader with
                test data compatible with model, optional as often the test loader can be
                generated from data_manager but is here for case when it hasn't been defined
                or there is a new test set.
        """
        self.model.train(self.train_loader, self.valid_loader)

        # use test loader in data_manager if there is one
        self.performances["full"] = self.model.test(self.test_loader if test_loader is None else test_loader)
        # make sure that theoretical best model is not stored
        self.model.current_model = None
        return self.performances["full"]

    def current_performance(self, test_loader: Optional[DataLoader] = None) -> Dict:
        if self.model.current_model is None:  # no AL steps taken so far
            self.model.train(self.l_loader, self.valid_loader)

        if test_loader is None:
            result = self.model.test(self.test_loader)
        else:
            result = self.model.test(test_loader)
        # reset current model, to avoid issues when model does not need to be trained during al_step.
        self.model.current_model = None
        return result

    @abstractmethod
    def active_learning_step(self, num_annotate: int) -> List[int]:
        """Implements a single step of the active learning strategy stopping and returning the
        unlabelled observations to be labelled as a list of dataset indices

        Args:
            num_annotate (int > 0): number of observations from u to suggest for labelling
        """
        pass

    def active_learning_update(self, indices: List[int], oracle_interface: object = None, update_tag: str = "") -> None:
        if oracle_interface is None:
            self.update_annotations(indices)
        else:
            oracle_interface.update_dataset(self.data_manager, indices)
            self.update_annotations(indices)

        # Logging
        self.iteration += 1
        logger.info("Length of labelled %s" % (len(self.l_indices)))
        logger.info("Length of unlabelled %s" % (len(self.u_indices)))
        logger.info("Percentage labelled %s" % self.percentage_labelled)
        self.log_labelled_by(indices, tag=update_tag)

    def full_active_learning_run(
        self,
        num_annotate: int,
        oracle_interface: object = None,
        test_loader: DataLoader = None,
    ) -> None:
        """Given the number of samples to annotate and a test loader
        this method will go through the entire active learning process of training
        the model on the labelled set, and recording the current performance
        based on this. Then it will proceed to compute uncertainties for the
        unlabelled observations, rank them, and get the top num_annotate observations
        labelled to be added to the next iteration's labelled dataset L'. This
        process repeats until there are no observations left in the unlabelled set.

        Args:
            num_annotate (int): number of observations to get annotated per iteration
            test_loader (pytorch.utils.data.DataLoader): test data with which we evaluate
                the current state of the model given the labelled set L
            oracle_interface (None or pyrelational.oracle_interface.OracleInterface)
        """

        while len(self.u_indices) > 0:

            # Obtain samples for labelling and pass to the oracle interface if supplied
            observations_for_labelling = self.active_learning_step(num_annotate)

            # Record the current performance
            self.performances[self.iteration] = self.current_performance(test_loader=test_loader)

            self.active_learning_update(
                observations_for_labelling,
                oracle_interface,
                update_tag=str(self.iteration),
            )

        # Final update the model and check final test performance
        self.model.train(self.l_loader, self.valid_loader)
        self.performances[self.iteration] = self.current_performance(test_loader=test_loader)

    def update_annotations(self, indices: List[int]) -> None:
        self.data_manager.update_train_labels(indices)

    def performance_history(self) -> pd.DataFrame:
        """Constructs a pandas table of performances of the model over the
        active learning iterations
        """
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
            row = [k]
            logger.info(self.performances[k])
            for c in columns[1:]:
                if isinstance(self.performances[k], list):
                    row.append(self.performances[k][0][c])
                else:
                    row.append(self.performances[k][c])
            df.append(row)

        pd_df = pd.DataFrame(df, columns=columns)
        return pd_df

    def log_labelled_by(self, indices: List[int], tag: str = ""):
        """Update the dictionary that records what the observation
        was labelled by. Default behaviour is to map observation to
        iteration at which it was labelled
        """
        for indx in indices:
            self.labelled_by[indx] = tag

    @property
    def u_indices(self) -> List[int]:
        return self.data_manager.u_indices

    @property
    def u_loader(self) -> DataLoader:
        return self.data_manager.get_unlabelled_loader()

    @property
    def l_indices(self) -> List[int]:
        return self.data_manager.l_indices

    @property
    def l_loader(self) -> DataLoader:
        return self.data_manager.get_labelled_loader()

    @property
    def train_loader(self) -> DataLoader:
        return self.data_manager.get_train_loader(full=True)

    @property
    def valid_loader(self) -> DataLoader:
        return self.data_manager.get_validation_loader()

    @property
    def test_loader(self) -> DataLoader:
        return self.data_manager.get_test_loader()

    @property
    def percentage_labelled(self) -> float:
        return self.data_manager.percentage_labelled()

    @property
    def dataset_size(self) -> int:
        return len(self.data_manager)

    def __repr__(self) -> str:
        return self.__class__.__name__

    def __str__(self) -> str:
        """Pretty prints contents inside ActiveLearningManager based on available attributes"""
        str_dm = str(self.data_manager)
        str_model = str(self.model)
        str_dataset_size = str(self.dataset_size)
        str_percentage_labelled = "%.3f" % self.percentage_labelled

        str_out = self.__repr__()
        str_out += "DataManager: %s \n" % str_dm
        str_out += "Model: %s \n" % str_model
        str_out += "Size of Dataset: %s \n" % str_dataset_size
        str_out += "Percentage of Dataset Labelled for Model: %s \n" % (str_percentage_labelled)
        if "full" in self.performances:
            str_out += "Theoretical performance: %s \n" % str(self.performances["full"])
        str_out += "Performance history \n"
        str_out += tabulate(self.performance_history(), tablefmt="pipe", headers="keys")

        return str_out
