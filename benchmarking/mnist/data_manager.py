"""Data manager for MNIST dataset.

We follow the setup in the BatchBald paper: https://arxiv.org/abs/1906.08158.
"""

from sklearn.model_selection import train_test_split

from pyrelational.data_managers import DataManager
from pyrelational.datasets.classification import MNIST


def get_mnist_datamanager(
    percentage_val: float = 0.1,
    labelled_size: int = 20,
    random_state: int = 42,
) -> DataManager:
    """Instantiate data manager for MNIST dataset.

    :param percentage_val: size in percentage of the validation split, defaults to 0.1
    :param labelled_size: number of initial labelled sample, defaults to 20
    :param random_state: random seed, defaults to 42
    :return: MNIST pyrelational data manager.
    """
    dataset = MNIST()
    train_ixs, test_ixs = dataset.data_splits[0]

    unlabelled_ixs, val_ixs = train_test_split(
        train_ixs,
        test_size=percentage_val,
        random_state=random_state,
        stratify=dataset.y[train_ixs],
    )

    labelled_ixs, unlabelled_ixs = train_test_split(
        unlabelled_ixs,
        train_size=labelled_size,
        random_state=random_state,
        stratify=dataset.y[unlabelled_ixs],
    )

    data_manager = DataManager(
        dataset=dataset,
        labelled_indices=labelled_ixs.tolist(),
        unlabelled_indices=unlabelled_ixs.tolist(),
        validation_indices=val_ixs.tolist(),
        test_indices=test_ixs.tolist(),
    )
    return data_manager
