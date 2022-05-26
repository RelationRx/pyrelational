.. _benchmark_datasets:

Benchmark datasets and AL task configurations
=============================================
A fundamental assumption in evaluating active learning strategies is that there exists a labelled subset of a training dataset that allows a model to perform as well (on the holdout test set) as using the entire training set. In evaluating an AL strategy we are interested in finding this subset efficiently, and maximising performance in an efficient manner.

To help users benchmark their strategies and active learning pipelines we have collected a range of datasets that have been used for benchmarking strategies in AL literature [#f1]_ . We provide classification and regression type datasets from a range of real world applications. Additionally we provide utilities to help create **cold** and **warm** start label initialisations corresponding to different active learning tasks to also help evaluate your strategy in these scenarios. More on these on the respective sections below.

This short tutorial will cover using the `datasets` subpackage containing classes that will download and process raw data into PyTorch Datasets that are ready for use with our DataManager classes. These extend completely standard PyTorch Dataset objects and can be used for normal ML experimentation as well. Each of the datasets will have additional parameters which describe the splitting of the dataset for cross-validation experiments, these are seeded for easier reproduction.

We hope that this resource helps make horizontal analysis of AL strategies across a range of datasets and
AL tasks easier. Better yet, lets hope it will garner interest in establishing a set of challening active learning benchmarks and tasks that can set a standard for the AL field.

Example usage: classification dataset
-------------------------------------

In this example we will look at the Wisconsin Breast Cancer (diagnostic) dataset [#f2]_ . It can be downloaded and processed with

.. code-block:: python

    from pyrelational.datasets import BreastCancerDataset
    dataset = BreastCancerDataset(n_splits = 5)

Where the `n_splits` argument specifies the number of train-test splits should be computed. For classification datasets the splits will be stratified by class. The `dataset` variable will behave like a regular PyTorch Dataset and is compatible with their excellent DataLoaders.

The `create_warm_start()` and `create_classification_cold_start()` functions in `pyrelational.datasets.benchmark_datamanager` will generate PyRelationAL DataManager objects corresponding to the following AL learning tasks inspired by Konyushkova et al. [#f3]_ .

- **Cold-start classification**: 1 observation for each class represented in the training set is labelled and the rest unlabeled.
- **Warm-start classification**: a randomly sampled 10 percent of the training set is labelled, the rest is unlabelled.

The following code snippet will return a DataManager corresponding to a cold-start initialisation for the breast cancer classification dataset using one of the precomputed splits:

.. code-block:: python

    from pyrelational.datasets import BreastCancerDataset
    dataset = BreastCancerDataset()
    train_indices = list(dataset.data_splits[0][0])
    test_indices = list(dataset.data_splits[0][1])
    dm = create_classification_cold_start(dataset, train_indices=train_indices, test_indices=test_indices)


Example usage: regression dataset
---------------------------------

This example will be identical to the classification case, except of course adjusted to be applicable for the regression ML task. We will use the UCI Diabetes dataset [#f4]_ . This can be downloaded and processed with

.. code-block:: python

    from pyrelational.datasets import DiabetesDataset
    dataset = DiabetesDataset(n_splits = 5)

As before the `n_splits` argument specifies the number of train-test splits that should be computed for the cross-validation setup. For regression these will be random splits, not stratified as in the classification case.

The `create_warm_start()` and `create_regression_cold_start()` functions in `pyrelational.datasets.benchmark_datamanager` will generate PyRelationAL DataManager objects corresponding to the following AL learning tasks inspired by Konyushkova et al. [#f3]_ .

- **Cold-start regression**: the two observations with highest euclidean pairwise distance in the train set are labelled, the rest is unlabelled.
- **Warm-start regression**: a randomly sampled 10 percent of the training set is labelled, the rest is unlabelled.

The following code snippet will return a DataManager corresponding to a cold-start initialisation for the diabetes regression dataset using one of the precomputed splits:

.. code-block:: python

    dataset = DiabetesDataset()
    train_indices = list(dataset.data_splits[0][0])
    test_indices = list(dataset.data_splits[0][1])
    dm = create_regression_cold_start(dataset, train_indices=train_indices, test_indices=test_indices)

We welcome any contributions to adding datasets and AL task configurations given they are justified by AL literature or make a convincing case for addition as a benchmark for AL strategies.

.. rubric:: Footnotes

.. [#f1] Please see the datasets API reference for a full listing
.. [#f2] https://archive.ics.uci.edu/ml/datasets/breast+cancer+wisconsin+(diagnostic)
.. [#f3] Learning Active Learning from Data from Konyushkova et al. NeurIPS 2017 (publicly available via https://arxiv.org/abs/1703.03365)
.. [#f4] https://archive.ics.uci.edu/ml/datasets/diabetes
