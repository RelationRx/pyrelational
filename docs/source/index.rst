.. PyRelationAL documentation master file, created by
   sphinx-quickstart on Thu Jun 17 15:33:16 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

:github_url: https://github.com/RelationRx/pyrelational

Welcome to PyRelationAL's documentation!
========================================

**PyRelationAL** is a python active learning library developed by `Relation Therapeutics <https://www.relationrx.com/>`_ for
rapidly implementing active learning pipelines from data management, model development (and Bayesian approximation), to creating novel active learning strategies.

.. toctree::
   :maxdepth: 1
   :caption: Notes

   notes/activelearning
   notes/installation
   notes/quick_start
   notes/using_your_own_data
   notes/using_the_model_api
   notes/using_your_own_strategy
   notes/benchmark_datasets

.. toctree::
   :glob:
   :maxdepth: 2
   :caption: Package modules

   reference/data.rst
   reference/datasets.rst
   reference/models.rst
   reference/informativeness.rst
   reference/strategies.rst

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


If the library is useful for your work please consider citing **PyRelationAL**.

.. code-block:: latex

   @misc{pyrelational,
         title={PyRelationAL},
         author={Relation Therapeutics},
         year={2021},
         publisher = {GitHub}
         journal = {GitHub repository}
         howpublished = {\url{https://github.com/RelationRx/pyrelational}}
   }
