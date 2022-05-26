# PyRelationAL

<p>
    <a alt="coverage">
        <img src="https://img.shields.io/badge/coverage-94%25-green" /></a>
    <a alt="semver">
        <img src="https://img.shields.io/badge/semver-0.1.6-blue" /></a>
    <a alt="documentation" href="https://pyrelational.readthedocs.io/en/latest/index.html">
        <img src="https://img.shields.io/badge/documentation-online-orange" /></a>
    <a alt="pypi" href="https://pypi.org/project/pyrelational/">
        <img src="https://img.shields.io/badge/pypi-online-yellow" /></a>
</p>

PyRelationAL is an open source Python library for the rapid and reliable construction of active learning (AL) pipelines and strategies. The toolkit offers a modular design for a flexible workflow that enables active learning with as little change to your models and datasets as possible. The package is primarily aimed at researchers so that they can rapidly reimplement, adapt, and create novel active learning strategies. For more information on how we achieve this you can consult the sections below, our comprehensive docs, or our paper. PyRelationAL is principally designed with PyTorch workflows in mind but can easily be extended to work with other ML frameworks.

Detailed in the **overview** section below, PyRelationAL offers:

- Data management in AL pipelines (*DataManager*)
- Wrappers for models to be used in AL workflows and strategies (*Model Manager*)
- (Optional) Ensembling and Bayesian inference approximation for point estimate models to quantifying uncertainty from point-estimate models (*Uncertainty estimation*).
- Active learning strategies and templates for making your own! (*Active learning strategy*)
- Benchmark datasets: an API for downloading datasets and AL task configurations based on literature for more standardised and painfree benchmarking.

One of our main incentives for making this library is to get more people interested in research and development of AL. Hence we have made primers, tutorials, and examples available on our website for newcomers (and experience AL practitioners alike). Experienced users can refer to our numerous examples to get started on their AL projects.

## Quick install

```bash
pip install pyrelational
```

## The `PyRelationAL` package

### Example

```python
# Active Learning package
import pyrelational as pal
from pyrelational.data import GenericDataManager
from pyrelational.strategies.generic_al_strategy import GenericActiveLearningStrategy
from pyrelational.models import GenericModel

# Instantiate data-loaders, models, trainers the usual Pytorch/PytorchLightning way
# In most cases, no change is needed to current workflow to incorporate
# active learning
data_manager = GenericDataManager(dataset, train_mask, validation_mask, test_mask)

# Create a model class that will handle model instantiation
model = GenericModel(ModelConstructor, model_config, trainer_config, **kwargs)

# Use the various implemented active learning strategies or define your own
al_manager = GenericActiveLearningStrategy(data_manager=data_manager, model=model)
al_manager.theoretical_performance(test_loader=test_loader)
al_manager.full_active_learning_run(num_annotate=100, test_loader=test_loader)
```

## Overview

![Overview](docs/images/active_learning_loop.png "Overview")

The `PyRelationAL` package decomposes the active learning workflow into four main components: 1) a **data manager**, 2) a **model**, 3) an **AL strategy** built around an informativeness function, and 4) an **oracle** (see Figure above). Note that the oracle is external to the package.

The **data manager** (defined in `pyrelational.data.data_manager.GenericDataManager`) wraps around a PyTorch Dataset and handles dataloader instantiation as well as tracking and updating of labelled and unlabelled sample pools.

The **model** (extending `pyrelational.models.generic_model.GenericModel`) wraps a user defined ML model (e.g. PyTorch Module, Flax module, or scikit-learn estimator) and handles instantiation, training, testing, as well as uncertainty quantification (e.g. ensembling, MC-dropout) if relevant. It also enables using ML models implemented using different ML frameworks (for example see `examples/demo/model_gaussianprocesses.py` or `examples/demo/scikit_estimator.py`).

The **AL strategy** (extending `pyrelational.strategies.generic_al_strategy.GenericActiveLearningStrategy`) defines an active learning strategy via an *informativeness measure* and a *query selection algorithm*. Together they compute the utility of a query or set of queries for a batch active mode strategy. We define various classic strategies for classification, regression, and task-agnostic scenarios based on the informativeness measures defined in `pyrelational.informativeness`. The flexible nature of the `GenericActiveLearningStrategy` allows for the construction of strategies from simple serial uncertainty sampling approaches to complex agents that leverage several informativeness measures, state and learning based query selection algorithms, with query batch building bandits under uncertainty from noisy oracles.

In addition to the main modules above we offer tools for **uncertainty estimation**. In recognition of the growing use of deep learning models we offer a suite of methods for Bayesian inference approximation to quantify uncertainty coming from the functional model such as MCDropout and ensembles of models (which may be used to also define query by committee and query by disagreement strategies).

Finally we to help test and benchmark strategies we offer **Benchmark datasets** and **AL task configurations**. We offer an API to a selection of datasets used previously in AL literature and offer each in several AL task configurations, such as cold and warm initialisations, for pain free benchmarking. For more details see our paper and documentation.

In the next section we briefly outline currently available strategies, informativeness measures, uncertainty estimation methods and some planned modules.

### List of included strategies and uncertainty estimation methods (constantly growing!)

#### Uncertainty Estimation

- MCDropout
- Ensemble of models (a.k.a. commitee)
- DropConnect (coming soon)
- SWAG (coming soon)
- MultiSWAG (coming soon)

#### Informativeness measures included in the library

##### Regression (N.B. PyRelationAL currently only supports single scalar regression tasks)

- Greedy score
- Least confidence score
- Expected improvement score
- Thompson sampling score
- Upper confidence bound (UCB) score
- BALD

##### Classification (N.B. PyRelationAL does not support multi-label classification at the moment)

- Least confidence
- Margin confidence
- Entropy based confidence
- Ratio based confidence
- BALD
- Thompson Sampling (coming soon)
- BatchBALD (coming soon)


##### Model agnostic and diversity sampling based approaches

- Representative sampling
- Diversity sampling
- Random acquisition
- BADGE

## Quickstart & examples
The `examples/` folder contains multiple scripts and notebooks demonstrating how to use PyRelationAL effectively.

The diverse examples scripts and notebooks aim to showcase how to use pyrelational in various scenario. Specifically,

- examples with regression
  - `lightning_diversity_regression.py`
  - `lightning_mixed_regression.py`
  - `mcdropout_uncertainty_regression.py`
  - `model_gaussianprocesses.py`
  - `model_badge.py`

- examples with classification tasks
  - `ensemble_uncertainty_classification.py`
  - `lightning_diversity_classification.py`
  - `lightning_representative_classification.py`
  - `mcdropout_uncertainty_classification.py`
  - `scikit_estimator.py`

- examples with task-agnostic acquisition
  - `lightning_diversity_classification.py`
  - `lightning_representative_classification.py`
  - `lightning_diversity_regression.py`
  - `model_badge.py`

- examples showcasing different uncertainty estimator
  - `ensemble_uncertainty_classification.py`
  - `mcdropout_uncertainty_classification.py`
  - `gpytorch_integration.py`
  - `model_badge.py`

- examples on how to create a custom acquisition strategy
  - `model_badge.py`
  - `lightning_mixed_regression.py`

- examples using different ML frameworks
  - `model_gaussianprocesses.py`
  - `scikit_estimator.py`


## Contributing to PyRelationAL

We welcome contributions to PyRelationAL, please see and adhere to the `CONTRIBUTING.md` and `CODE_OF_CONDUCT.md` guidelines.

### Prerequisites and setup

For those just using the package, installation only requires standard ML packages and PyTorch. More advanced users or those wishing to contribute should start with a new virtual environment (miniconda environment recommended) and install standard learning packages and numerical tools.

```bash
pip install -r requirements.txt
```

If you wish to contribute to the code, run `pre-commit install` after the above step.

### Organisation of repository

- `pyrelational` folder contains the source code for the PyRelationAL package. It contains the main sub-packages for active learning strategies, various informativeness measures, and methods for estimating posterior uncertainties.
- `examples` folder contains various example scripts and notebooks detailing how the package can be used to construct novel strategies, work with different ML frameworks, and use your own data
- `tests` folder contains unit tests for pyrelational package
- `docs` folder contains documentation and assets for docs

### Building the docs

Make sure you have `sphinx` and `sphinx-rtd-theme` packages installed (`pip install sphinx sphinx_rtd_theme` will install this).

To generate the docs, `cd` into the `docs/` directory and run `make html`. This will generate the docs
at `docs/_build/html/index.html`.
