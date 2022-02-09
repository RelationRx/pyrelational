"""
Simple datasets in PyTorch to use in examples
"""
import torch
from sklearn.datasets import load_breast_cancer, load_diabetes
from torch.utils.data import Dataset


class DiabetesDataset(Dataset):
    """A small regression dataset for examples"""

    def __init__(self):
        # Load the diabetes dataset
        diabetes_X, diabetes_y = load_diabetes(return_X_y=True)
        self.x = torch.FloatTensor(diabetes_X)
        self.y = torch.FloatTensor(diabetes_y)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class BreastCancerDataset(Dataset):
    """A small classification dataset for examples"""

    def __init__(self):
        super(BreastCancerDataset, self).__init__()
        sk_x, sk_y = load_breast_cancer(return_X_y=True)
        self.x = torch.FloatTensor(sk_x)
        self.y = torch.LongTensor(sk_y)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
