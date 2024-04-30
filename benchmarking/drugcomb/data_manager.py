from pyrelational.data_managers import DataManager
from pyrelational.datasets import DrugCombDataset


class DrugCombDataManager(DataManager):

    def __init__(self, seed: int, batch_size: int = 32, num_workers: int = 0):

        dataset = DrugCombDataset()
        train_indices, test_indices = dataset.data_splits[0]
        super().__init__(
            dataset,
            train_indices=train_indices.tolist(),
            test_indices=test_indices.tolist(),
            random_label_size=256,
            hit_ratio_at=5,
            random_seed=seed,
            loader_batch_size=batch_size,
            loader_num_workers=num_workers,
        )
