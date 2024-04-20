from typing import Any, Generator, Sequence

import torch
import torch.distributions as distributions

from pyrelational.datasets.base import BaseDataset

from .utils import create_splits


class SynthClass1(BaseDataset):
    """
    Synth1 dataset generates samples from two Gaussian distributions, representing two classes.

    :param size: Total number of samples in the dataset.
    :param n_splits: Number of stratified splits for the dataset.
    :param random_seed: Seed for random number generator for reproducibility.
    """

    def __init__(self, size: int = 500, n_splits: int = 5, random_seed: int = 1234):
        super().__init__(n_splits=n_splits, random_seed=random_seed)
        self.size = size
        self._create_data()

    def _create_data(self) -> None:
        pos_distribution = distributions.MultivariateNormal(torch.tensor([1, 1], dtype=torch.float), torch.eye(2))
        neg_distribution = distributions.MultivariateNormal(torch.tensor([-1, -1], dtype=torch.float), torch.eye(2))
        num_pos = self.size // 2
        pos_samples = pos_distribution.sample(sample_shape=torch.Size([num_pos]))
        neg_samples = neg_distribution.sample(sample_shape=torch.Size([self.size - num_pos]))

        self.x = torch.cat([pos_samples, neg_samples])
        self.y = torch.cat([torch.ones(num_pos, dtype=torch.long), torch.zeros(self.size - num_pos, dtype=torch.long)])
        self.data_splits = create_splits(self.x, self.y, self.n_splits, self.random_seed)


class SynthClass2(BaseDataset):
    """
    Synth2 dataset creates a more complex synthetic dataset from six Gaussian blobs divided into two classes.

    :param size: Total number of samples in the dataset.
    :param n_splits: Number of stratified splits for the dataset.
    :param random_seed: Seed for random number generator for reproducibility.
    """

    def __init__(self, size: int = 500, n_splits: int = 5, random_seed: int = 1234):
        super().__init__(n_splits=n_splits, random_seed=random_seed)
        self.size = size
        self._create_data()

    def _create_data(self) -> None:
        distributions_params = [
            ([0, 5], torch.eye(2)),
            ([0, -5], torch.eye(2)),
            ([-5, 10], torch.eye(2)),
            ([5, 10], torch.eye(2)),
            ([-5, -10], torch.eye(2)),
            ([5, -10], torch.eye(2)),
        ]
        class_samples = [self.size // 6 + (1 if i < self.size % 6 else 0) for i in range(6)]

        samples = []
        targets = []
        for i, ((mean, cov), num_samples) in enumerate(zip(distributions_params, class_samples)):
            dist = distributions.MultivariateNormal(torch.tensor(mean, dtype=torch.float), cov)
            samples.append(dist.sample(sample_shape=torch.Size([num_samples])))
            targets.append(
                torch.full((num_samples,), i // 3, dtype=torch.long)
            )  # 0 for the first three, 1 for the last three

        self.x = torch.cat(samples)
        self.y = torch.cat(targets)
        self.data_splits = create_splits(self.x, self.y, self.n_splits, self.random_seed)


class SynthClass3(BaseDataset):
    """
    SynthClass3 generates data using multivariate Gaussian distributions with different covariance matrices.

    :param size: Total number of samples in the dataset.
    :param n_splits: Number of stratified splits for the dataset.
    :param random_seed: Seed for random number generator for reproducibility.
    """

    def __init__(self, size: int = 500, n_splits: int = 5, random_seed: int = 1234):
        super().__init__(n_splits=n_splits, random_seed=random_seed)
        self.size = size
        self._create_data()

    def _create_data(self) -> None:
        cov = torch.FloatTensor([[0.60834549, -0.63667341], [-0.40887718, 0.85253229]])
        cov = torch.matmul(cov, cov.T)

        pos_dist_1 = distributions.MultivariateNormal(torch.FloatTensor([0, 0]), cov)
        pos_dist_2 = distributions.MultivariateNormal(torch.FloatTensor([3, 10]), cov)
        neg_dist_1 = distributions.MultivariateNormal(torch.FloatTensor([3, 3]), torch.FloatTensor([[1, 2], [2, 7]]))

        num_pos = int(self.size / 2.0)
        num_neg1 = self.size - num_pos
        num_pos1, num_pos2 = [len(x) for x in self._split(range(num_pos), 2)]

        pos_samples_1 = torch.vstack([pos_dist_1.sample() for _ in range(num_pos1)])
        pos_samples_2 = torch.vstack([pos_dist_2.sample() for _ in range(num_pos2)])
        neg_samples_1 = torch.vstack([neg_dist_1.sample() for _ in range(num_neg1)])

        pos_targets = torch.ones(num_pos, dtype=torch.long)
        neg_targets = torch.ones(num_neg1, dtype=torch.long) * 0

        self.x = torch.cat([pos_samples_1, pos_samples_2, neg_samples_1])
        self.y = torch.cat([pos_targets, neg_targets])
        self.data_splits = create_splits(self.x, self.y, self.n_splits, self.random_seed)

    @staticmethod
    def _split(iterable: Sequence[Any], n: int) -> Generator[Sequence[Any], None, None]:
        # split the iterable into n approximately same size parts
        k, m = divmod(len(iterable), n)
        return (iterable[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(n))
