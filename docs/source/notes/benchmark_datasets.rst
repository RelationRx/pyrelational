.. _benchmark_datasets:

Benchmarking Active Learning
============================
A fundamental assumption in evaluating active learning strategies is that there exists a labelled subset of a training dataset that allows a model to perform as well (on the holdout test set) as using the entire training set. In evaluating an AL strategy we are interested in finding this subset efficiently, and maximising performance in an efficient manner.

To help users benchmark their strategies and active learning pipelines we have collected a range of datasets that have been used for benchmarking strategies in AL literature [#f1]_ . We provide classification and regression type datasets from a range of real world applications. Additionally we provide utilities to help create **cold** and **warm** start label initialisations corresponding to different active learning tasks to also help evaluate your strategy in these scenarios. More on these on the respective sections below.

This short tutorial will cover using the `datasets` subpackage containing classes that will download and process raw data into PyTorch Datasets that are ready for use with our DataManager classes. These extend completely standard PyTorch Dataset objects and can be used for normal ML experimentation as well. Each of the datasets will have additional parameters which describe the splitting of the dataset for cross-validation experiments, these are seeded for easier reproduction.

We hope that this resource helps make horizontal analysis of AL strategies across a range of datasets and
AL tasks easier. Better yet, lets hope it will garner interest in establishing a set of challening active learning benchmarks and tasks that can set a standard for the AL field.

Example usage: classification dataset
=====================================

The following

For classification


Example usage: regression dataset
=================================


.. rubric:: Footnotes

.. [#f1] Please see the datasets API reference for a full listing
