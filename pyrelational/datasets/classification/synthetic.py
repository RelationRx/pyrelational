import torch
import torch.distributions as distributions

from .base import BaseDataset


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
        self._create_splits()


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
        self._create_splits()


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
        pos_dist_1 = distributions.MultivariateNormal(torch.tensor([0, 0], dtype=torch.float), torch.eye(2))
        pos_dist_2 = distributions.MultivariateNormal(torch.tensor([3, 10], dtype=torch.float), torch.eye(2))
        neg_dist = distributions.MultivariateNormal(
            torch.tensor([3, 3], dtype=torch.float), torch.tensor([[1, 2], [2, 7]], dtype=torch.float)
        )

        num_pos = self.size // 2
        num_neg = self.size - num_pos
        num_pos1, num_pos2 = num_pos // 2, num_pos - num_pos // 2

        pos_samples_1 = pos_dist_1.sample(sample_shape=torch.Size([num_pos1]))
        pos_samples_2 = pos_dist_2.sample(sample_shape=torch.Size([num_pos2]))
        neg_samples = neg_dist.sample(sample_shape=torch.Size([num_neg]))

        self.x = torch.cat([pos_samples_1, pos_samples_2, neg_samples])
        self.y = torch.cat([torch.ones(num_pos, dtype=torch.long), torch.zeros(num_neg, dtype=torch.long)])
        self._create_splits()
