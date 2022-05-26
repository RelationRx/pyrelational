pyrelational.datasets
=====================


Classification datasets
-----------------------

The following classes contain a variety of classic classification datasets that have been used in different active learning papers. Each behaves the same as a PyTorch Dataset.

.. automodule:: pyrelational.datasets.classification
   :members:
   :undoc-members:
   :show-inheritance:

Regression datasets
-----------------------

The following classes contain a variety of classic regression datasets that have been used in different active learning papers. Each behaves the same as a PyTorch Dataset.

.. automodule:: pyrelational.datasets.regression
   :members:
   :undoc-members:
   :show-inheritance:


Benchmark DataManager
---------------------

The following functions accept the datasets defined in this package to produce DataManagers containing labelling initialisations that correspond to cold and warm start active learning tasks. These can be used for benchmarking strategies quickly.

.. automodule:: pyrelational.datasets.benchmark_datamanager
   :members:
   :undoc-members:
   :show-inheritance:
