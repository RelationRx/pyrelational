"""
Class to help download and do the initial processing of dataset on the UCI database
"""

import logging
import os
import urllib.request
import zipfile
from os import path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold, StratifiedKFold


class UCIDatasets:
    def __init__(self, name, data_dir="/tmp/", n_splits=10):
        self.datasets = {
            "concrete": "https://archive.ics.uci.edu/ml/machine-learning-databa"
            + "ses/concrete/compressive/Concrete_Data.xls",
            "energy": "http://archive.ics.uci.edu/ml/machine-learning-databases/00242/ENB2012_data.xlsx",
            "power": "https://archive.ics.uci.edu/ml/machine-learning-databases/00294/CCPP.zip",
            "wine": "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv",
            "yacht": "http://archive.ics.uci.edu/ml/machine-learning-databases/00243/yacht_hydrodynamics.data",
            "glass": "https://archive.ics.uci.edu/ml/machine-learning-databases/glass/glass.data",
            "parkinsons": "https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data",
            "seeds": "https://archive.ics.uci.edu/ml/machine-learning-databases/00236/seeds_dataset.txt",
            "airfoil": "https://archive.ics.uci.edu/ml/machine-learning-databases/00291/airfoil_self_noise.dat",
        }
        self.data_dir = data_dir
        self.name = name
        self.n_splits = n_splits

        # flag for classification/regression dataset
        if name in ["glass", "parkinsons", "seeds"]:
            self.classification = True
        else:
            self.classification = False

        self._load_dataset()

    def _load_dataset(self):
        if self.name not in self.datasets:
            raise Exception("Not part of datasets supported in PyRelationAL at the moment")
        if not path.exists(self.data_dir):
            os.mkdir(self.data_dir)
        if not path.exists(self.data_dir + "UCI"):
            os.mkdir(self.data_dir + "UCI")

        url = self.datasets[self.name]
        file_name = url.split("/")[-1]
        if not path.exists(self.data_dir + "UCI/" + file_name):
            urllib.request.urlretrieve(self.datasets[self.name], self.data_dir + "UCI/" + file_name)
        data = None

        if self.name == "housing":
            data = pd.read_csv(self.data_dir + "UCI/housing.data", header=0, delimiter=r"\s+").values
            self.data = data[np.random.permutation(np.arange(len(data)))]

        elif self.name == "airfoil":
            data = pd.read_csv(self.data_dir + "UCI/airfoil_self_noise.dat", header=0, delimiter=r"\s+").values
            self.data = data[np.random.permutation(np.arange(len(data)))]

        elif self.name == "concrete":
            data = pd.read_excel(self.data_dir + "UCI/Concrete_Data.xls", header=0).values
            self.data = data[np.random.permutation(np.arange(len(data)))]

        elif self.name == "energy":
            data = pd.read_excel(self.data_dir + "UCI/ENB2012_data.xlsx", header=0).values
            self.data = data[np.random.permutation(np.arange(len(data)))]

        elif self.name == "power":
            zipfile.ZipFile(self.data_dir + "UCI/CCPP.zip").extractall(self.data_dir + "UCI/")
            data = pd.read_excel(self.data_dir + "UCI/CCPP/Folds5x2_pp.xlsx", header=0).values
            self.data = data

        elif self.name == "wine":
            data = pd.read_csv(self.data_dir + "UCI/winequality-red.csv", header=1, delimiter=";").values
            self.data = data[np.random.permutation(np.arange(len(data)))]

        elif self.name == "yacht":
            data = pd.read_csv(self.data_dir + "UCI/yacht_hydrodynamics.data", header=1, delimiter=r"\s+").values
            self.data = data[np.random.permutation(np.arange(len(data)))]

        elif self.name == "glass":
            data = pd.read_csv(self.data_dir + "UCI/glass.data", delimiter=",").values
            self.data = data[np.random.permutation(np.arange(len(data)))]

        elif self.name == "parkinsons":
            data = pd.read_csv(self.data_dir + "UCI/parkinsons.data", header=0, delimiter=",", index_col=0)
            # we need to get the "status" column as the target
            columns = list(data.columns)
            columns = set(columns) - set(["status"])
            reordered_columns = list(columns)
            reordered_columns.extend(["status"])
            data = data[reordered_columns]
            data = data.values
            self.data = data[np.random.permutation(np.arange(len(data)))]

        elif self.name == "seeds":
            data = pd.read_csv(self.data_dir + "UCI/seeds_dataset.txt", delimiter=r"\s+", engine="python").values
            self.data = data[np.random.permutation(np.arange(len(data)))]

        self.in_dim = data.shape[1] - 1
        self.out_dim = 1

        if self.classification:
            x, y = self.data[:, : self.in_dim], self.data[:, self.in_dim :]
            skf = StratifiedKFold(n_splits=self.n_splits)
            self.data_splits = skf.split(x, y)
            self.data_splits = [(idx[0], idx[1]) for idx in self.data_splits]
        else:
            kf = KFold(n_splits=self.n_splits)
            self.data_splits = kf.split(data)
            self.data_splits = [(idx[0], idx[1]) for idx in self.data_splits]

    def get_split(self, split=-1, train=True):
        if split == -1:
            split = 0
        if 0 <= split and split < self.n_splits:
            train_index, test_index = self.data_splits[split]
            x_train, y_train = self.data[train_index, : self.in_dim], self.data[train_index, self.in_dim :]
            x_test, y_test = self.data[test_index, : self.in_dim], self.data[test_index, self.in_dim :]
            x_means, x_stds = x_train.mean(axis=0), x_train.var(axis=0) ** 0.5
            y_means, y_stds = y_train.mean(axis=0), y_train.var(axis=0) ** 0.5
            x_train = (x_train - x_means) / x_stds
            y_train = (y_train - y_means) / y_stds
            x_test = (x_test - x_means) / x_stds
            y_test = (y_test - y_means) / y_stds
            if train:
                inps = torch.from_numpy(x_train).float()
                tgts = torch.from_numpy(y_train).float()
                train_data = torch.utils.data.TensorDataset(inps, tgts)
                return train_data
            else:
                inps = torch.from_numpy(x_test).float()
                tgts = torch.from_numpy(y_test).float()
                test_data = torch.utils.data.TensorDataset(inps, tgts)
                return test_data

    def get_full_split(self, split=-1):
        """Returns a single dataset with the test observations stacked on the train observations"""
        if split == -1:
            split = 0
        if 0 <= split and split < self.n_splits:
            train_index, test_index = self.data_splits[split]
            x_train, y_train = self.data[train_index, : self.in_dim], self.data[train_index, self.in_dim :]
            x_test, y_test = self.data[test_index, : self.in_dim], self.data[test_index, self.in_dim :]

            if self.classification:
                # standardize
                x_means, x_stds = x_train.mean(axis=0), x_train.var(axis=0) ** 0.5
                x_train = (x_train - x_means) / x_stds
                x_test = (x_test - x_means) / x_stds

                inps = torch.from_numpy(x_train).float()
                tgts = torch.from_numpy(y_train).long()
                inps_test = torch.from_numpy(x_test).float()
                tgts_test = torch.from_numpy(y_test).long()
            else:
                # standardize
                x_means, x_stds = x_train.mean(axis=0), x_train.var(axis=0) ** 0.5
                y_means, y_stds = y_train.mean(axis=0), y_train.var(axis=0) ** 0.5
                x_train = (x_train - x_means) / x_stds
                y_train = (y_train - y_means) / y_stds
                x_test = (x_test - x_means) / x_stds
                y_test = (y_test - y_means) / y_stds

                inps = torch.from_numpy(x_train).float()
                tgts = torch.from_numpy(y_train).float()
                inps_test = torch.from_numpy(x_test).float()
                tgts_test = torch.from_numpy(y_test).float()

            inps = torch.cat([inps, inps_test])
            tgts = torch.cat([tgts, tgts_test])

            dataset = torch.utils.data.TensorDataset(inps, tgts)
            return dataset

    def get_simple_dataset(self):
        """Simply return the dataset so that we can apply the splits on top after"""
        x, y = self.data[:, : self.in_dim], self.data[:, self.in_dim :]

        if self.classification:
            inps = torch.from_numpy(x).float()
            tgts = torch.from_numpy(y).long()
        else:
            inps = torch.from_numpy(x).float()
            tgts = torch.from_numpy(y).float()

        dataset = torch.utils.data.TensorDataset(inps, tgts)
        return dataset
