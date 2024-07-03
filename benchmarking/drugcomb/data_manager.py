from pyrelational.data_managers import DataManager
from pyrelational.datasets import DrugCombDataset


class DrugCombDataManager(DataManager):

    def __init__(
        self,
        seed: int,
        batch_size: int = 32,
        num_workers: int = 0,
        hit_ratio_at: int | float = 5,
        initial_labelled_size: int | float = 0.2,
    ):

        dataset = DrugCombDataset()
        train_indices, test_indices = dataset.data_splits[0]
        super().__init__(
            dataset,
            train_indices=train_indices.tolist(),
            test_indices=test_indices.tolist(),
            random_label_size=initial_labelled_size,
            hit_ratio_at=hit_ratio_at,
            random_seed=seed,
            loader_batch_size=batch_size,
            loader_num_workers=num_workers,
        )
