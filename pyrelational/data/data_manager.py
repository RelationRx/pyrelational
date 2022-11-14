import logging
import random
import warnings
from typing import (
    Any,
    Callable,
    Collection,
    Iterable,
    List,
    Optional,
    Protocol,
    Sequence,
    Sized,
    Tuple,
    TypeVar,
    Union,
    cast,
)

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler, Subset

T = TypeVar("T")
logger = logging.getLogger()


class SizedDataset(Dataset, Sized):
    ...


class GenericDataManager(object):
    def __init__(
        self,
        dataset: Dataset,
        train_indices: Optional[List[int]] = None,
        labelled_indices: Optional[List[int]] = None,
        unlabelled_indices: Optional[List[int]] = None,
        validation_indices: Optional[List[int]] = None,
        test_indices: Optional[List[int]] = None,
        random_label_size: Union[float, int] = 0.1,
        hit_ratio_at: Optional[Union[int, float]] = None,
        random_seed: int = 1234,
        loader_batch_size: Union[int, str] = 1,
        loader_shuffle: bool = False,
        loader_sampler: Optional[Sampler[int]] = None,
        loader_batch_sampler: Optional[Sampler[Sequence[int]]] = None,
        loader_num_workers: int = 0,
        loader_collate_fn: Optional[Callable[[List[T]], Any]] = None,
        loader_pin_memory: bool = False,
        loader_drop_last: bool = False,
        loader_timeout: float = 0,
    ):
        """
        DataManager for active learning pipelines

        :param dataset: A PyTorch dataset whose indices refer to individual samples of study
        :param train_indices: An iterable of indices mapping to training sample indices in the dataset
        :param labelled_indices: An iterable of indices  mapping to labelled training samples
        :param unlabelled_indices: An iterable of indices to unlabelled observations in the dataset
        :param validation_indices: An iterable of indices to observations used for model validation
        :param test_indices: An iterable of indices to observations in the input dataset used for
        test performance of the model
        :param random_label_size: Only used when labelled and unlabelled indices are not provided. Sets the size of
        labelled set (should either be the number of samples or ratio w.r.t. train set)
        :param hit_ratio_at: optional argument setting the top percentage threshold to compute hit ratio metric
        :param random_seed: random seed
        :param loader_batch_size: batch size for dataloader
        :param loader_shuffle: shuffle flag for dataloader
        :param loader_sampler: a sampler for the dataloaders
        :param loader_batch_sampler: a batch sampler for the dataloaders
        :param loader_num_workers: number of cpu workers for dataloaders
        :param loader_collate_fn: collate fn for dataloaders
        :param loader_pin_memory: pin memory flag for dataloaders
        :param loader_drop_last: drop last flag for dataloaders
        :param loader_timeout: timeout value for dataloaders
        """
        super(GenericDataManager, self).__init__()
        dataset = self._check_dataset_has_len(dataset)

        self.dataset = dataset

        # Loader specific arguments
        self.loader_batch_size = loader_batch_size
        self.loader_shuffle = loader_shuffle
        self.loader_sampler = loader_sampler
        self.loader_batch_sampler = loader_batch_sampler
        self.loader_num_workers = loader_num_workers
        self.loader_collate_fn = loader_collate_fn
        self.loader_pin_memory = loader_pin_memory
        self.loader_drop_last = loader_drop_last
        self.loader_timeout = loader_timeout

        # Resolve masks and the values they should take given inputs
        self._resolve_dataset_split_indices(train_indices, validation_indices, test_indices)

        # Set l and u indices according to mask arguments
        # and need to check that they arent part of
        if labelled_indices is not None:
            if unlabelled_indices is not None:
                self._ensure_no_l_u_intersection(labelled_indices, unlabelled_indices)
            else:
                unlabelled_indices = list(set(self.train_indices) - set(labelled_indices))
            self.labelled_indices = labelled_indices
            self.l_indices = labelled_indices
            self.unlabelled_indices = unlabelled_indices
            self.u_indices = unlabelled_indices
        else:
            logger.info("## Labelled and/or unlabelled mask unspecified")
            self.random_label_size = random_label_size
            self.process_random(random_seed)
        self._ensure_no_l_or_u_leaks()
        self._top_unlabelled_set(hit_ratio_at)

    def _ensure_no_split_leaks(
        self,
        train_indices: List[int],
        validation_indices: Optional[List[int]],
        test_indices: List[int],
    ) -> None:
        tt = set.intersection(set(train_indices), set(test_indices))
        tv, vt = None, None
        if validation_indices is not None:
            tv = set.intersection(set(train_indices), set(validation_indices))
            vt = set.intersection(set(validation_indices), set(test_indices))
        if tv or tt or vt:
            raise ValueError("There is an overlap between the split indices supplied")

    def _ensure_no_empty_train(self, train_indices: List[int]) -> None:
        """ensures that the train set is not empty, as there is no need to
        do anything if its empty
        """
        if len(train_indices) == 0:
            raise ValueError("The train set is empty")

    def _ensure_no_l_u_intersection(self, labelled_indices: List[int], unlabelled_indices: List[int]):
        if set.intersection(set(labelled_indices), set(unlabelled_indices)):
            raise ValueError("There is overlap between labelled and unlabelled samples")

    def _ensure_no_l_or_u_leaks(self) -> None:
        """ensures that there are no leaks of labelled or unlabelled indices
        in the validation or tests indices
        """
        if self.validation_indices is not None:
            v_overlap = set.intersection(set(self.l_indices), set(self.validation_indices))
            if v_overlap:
                raise ValueError(
                    f"There is {len(v_overlap)} sample overlap between the labelled indices and the validation set"
                )
            v_overlap = set.intersection(set(self.u_indices), set(self.validation_indices))
            if v_overlap:
                raise ValueError(
                    f"There is {len(v_overlap)} sample overlap between the unlabelled indices and the validation set"
                )

        if self.test_indices is not None:
            t_overlap = set.intersection(set(self.l_indices), set(self.test_indices))
            if t_overlap:
                raise ValueError(
                    f"There is {len(t_overlap)} sample overlap between the labelled indices and the test set"
                )

            # save memory by using same variables
            t_overlap = set.intersection(set(self.u_indices), set(self.test_indices))
            if t_overlap:
                raise ValueError(
                    f"There is {len(t_overlap)} sample overlap between the unlabelled indices and the test set"
                )

    def _resolve_dataset_split_indices(
        self,
        train_indices: Optional[List[int]],
        validation_indices: Optional[List[int]],
        test_indices: Optional[List[int]],
    ) -> None:
        """This function is used to resolve what values the indices should be given
        when only a partial subset of them is supplied

        .. image:: docs/images/data_indices_diagram.png
        

        :param train_indices: list of indices in dataset for train set
        :param validation_indices: list of indices in dataset for validation set
        :param test_indices: list of indices in dataset for test set
        """

        remaining_indices = set(range(len(self.dataset))) - set.union(
            set(train_indices if train_indices is not None else []),
            set(validation_indices if validation_indices is not None else []),
            set(test_indices if test_indices is not None else []),
        )

        if train_indices is None:
            if test_indices is None:
                raise ValueError("No train or test specified, too ambigious to set values")
            train_indices = list(remaining_indices)
        elif test_indices is None:
            test_indices = list(remaining_indices)
        elif remaining_indices:
            warnings.warn(f"{len(remaining_indices)} indices are not found in any split")

        self._ensure_no_empty_train(train_indices)
        self._ensure_no_split_leaks(train_indices, validation_indices, test_indices)
        self.train_indices = train_indices
        self.validation_indices = validation_indices
        self.test_indices = test_indices

    def __len__(self) -> int:
        # Override this if necessary
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor]:
        # So that one can access samples by index directly
        return self.dataset[idx]

    def _top_unlabelled_set(self, percentage: Optional[Union[int, float]] = None) -> None:
        if percentage is None:
            self.top_unlabelled = None
        else:
            if isinstance(percentage, int):
                percentage /= 100
            assert 0 < percentage < 1, "hit ratio's percentage should be strictly between 0 and 1 (or 0 and 100)"
            ixs = np.array(self.u_indices)
            percentage = int(percentage * len(ixs))
            y = torch.stack(self.get_sample_labels(ixs)).squeeze()
            threshold = np.sort(y.abs())[-percentage]
            self.top_unlabelled = set(ixs[(y.abs() >= threshold).numpy().astype(bool)])

    def get_train_set(self) -> Dataset:
        train_subset = Subset(self.dataset, self.train_indices)
        return train_subset

    def get_validation_set(self) -> Optional[Subset]:
        if self.validation_indices is None:
            return None
        validation_subset = Subset(self.dataset, self.validation_indices)
        return validation_subset

    def get_test_set(self) -> Subset:
        test_subset = Subset(self.dataset, self.test_indices)
        return test_subset

    def get_train_loader(self, full: bool = False) -> DataLoader:
        if full:
            # return full training set with unlabelled included (for strategy evaluation)
            train_loader = self.create_loader(Subset(self.dataset, (self.l_indices + self.u_indices)))
            return train_loader
        else:
            return self.get_labelled_loader()

    def get_validation_loader(self) -> Optional[DataLoader]:
        validation_set = self.get_validation_set()
        if validation_set is None:
            return None
        return self.create_loader(validation_set)

    def get_test_loader(self) -> DataLoader:
        return self.create_loader(self.get_test_set())

    def get_unlabelled_loader(self) -> DataLoader:
        return self.create_loader(Subset(self.dataset, self.u_indices))

    def get_labelled_loader(self) -> DataLoader:
        return self.create_loader(Subset(self.dataset, self.l_indices))

    def process_random(self, seed=0) -> None:
        """Processes the dataset to produce a random subsets of labelled and unlabelled
        samples from the dataset based on the ratio given at initialisation and creates
        the data_loaders
        """
        if isinstance(self.random_label_size, float):
            assert 0 < self.random_label_size < 1, "if a float, random_label_size should be between 0 and 1"
            num_labelled = int(self.random_label_size * len(self.train_indices))
        else:
            num_labelled = self.random_label_size

        logger.info("## Randomly generating labelled subset with {} samples from the train data".format(num_labelled))
        random.seed(seed)
        l_indices = set(random.sample(self.train_indices, num_labelled))
        u_indices = set(self.train_indices) - set(l_indices)

        self.l_indices = list(l_indices)
        self.u_indices = list(u_indices)

    def update_train_labels(self, indices: List[int]) -> None:
        """Updates the L and U sets of the dataset

        Different behaviour based on whether this is done in evaluation mode or real mode.
        The difference is that in evaluation mode the dataset already has the label, so it
        is a matter of making sure the observations are moved from the unlabelled set to the
        labelled set.
        """
        self.l_indices = list(set(self.l_indices + indices))
        self.u_indices = list(set(self.u_indices) - set(indices))

    def percentage_labelled(self) -> float:
        total_len = len(self.l_indices) + len(self.u_indices)
        num_labelled = len(self.l_indices)
        return num_labelled / float(total_len)

    def get_sample(self, ds_index: int) -> Tuple[torch.Tensor]:
        return self[ds_index]

    def get_sample_feature_vector(self, ds_index: int) -> torch.Tensor:
        """To be reviewed for deprecation (for datasets without tensors)"""
        sample = self.get_sample(ds_index)
        sample = sample[0].flatten()
        return sample

    def get_sample_feature_vectors(self, ds_indices: List[int]) -> List[torch.Tensor]:
        """To be reviewed for deprecation (for datasets without tensors)"""
        res = []
        for ds_index in ds_indices:
            res.append(self.get_sample_feature_vector(ds_index))
        return res

    def get_sample_labels(self, ds_indices: Collection[int]) -> List[torch.Tensor]:
        res = []
        for ds_index in ds_indices:
            res.append(self[ds_index][-1])  # assumes labels are last in output of dataset
        return res

    def create_loader(self, dataset: Dataset) -> DataLoader:
        """Utility to help create dataloader with specifications set at initialisation"""
        dataset = self._check_dataset_has_len(dataset)
        batch_size = self.loader_batch_size if isinstance(self.loader_batch_size, int) else len(dataset)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=self.loader_shuffle,
            sampler=self.loader_sampler,
            batch_sampler=self.loader_batch_sampler,
            num_workers=self.loader_num_workers,
            collate_fn=self.loader_collate_fn,
            pin_memory=self.loader_pin_memory,
            drop_last=self.loader_drop_last,
            timeout=self.loader_timeout,
        )
        return loader

    def __repr__(self) -> str:
        return self.__class__.__name__

    def __str__(self) -> str:
        """Pretty print a summary of the data_manager contents"""
        str_percentage_labelled = "%.3f" % (self.percentage_labelled())
        str_out = self.__repr__()
        if self.train_indices is not None:
            str_out += "\nTraining set size: {}\n".format(len(self.train_indices))
        if self.labelled_indices is not None:
            str_out += "Labelled: {}, Unlabelled: {}\n".format(len(self.labelled_indices), len(self.unlabelled_indices))
        str_out += "Percentage Labelled: {}".format(str_percentage_labelled)

        return str_out

    @staticmethod
    def _check_dataset_has_len(dataset: Dataset) -> SizedDataset:
        """Check Dataset is Sized (has a __len__ method)"""
        if not isinstance(dataset, Sized):
            raise AttributeError("dataset must have __len__ method defined")
        return cast(SizedDataset, dataset)
