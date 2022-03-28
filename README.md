# PyRelationAL


<p>
    <a alt="coverage">
        <img src="https://img.shields.io/badge/coverage-93%25-green" /></a>
    <a alt="semver">
        <img src="https://img.shields.io/badge/semver-0.1.5-blue" /></a>
    <a alt="documentation" href="https://pyrelational.readthedocs.io/en/latest/index.html">
        <img src="https://img.shields.io/badge/documentation-online-orange" /></a>
    <a alt="pypi" href="https://pypi.org/project/pyrelational/">
        <img src="https://img.shields.io/badge/pypi-online-yellow" /></a>
</p>

### Quick install

`pip install pyrelational`

### Organisation of repository

- `pyrelational` folder contains the source code for the PyRelationAL package. It contains the main sub-packages for active learning strategies, various informativeness measures, and methods for estimating posterior uncertainties.
- `examples` folder contains various example scripts and notebooks detailing how the package can be used
- `tests` folder contains unit tests for pyrelational package
- `docs` folder contains docs and assets for docs

### The `PyRelationAL` package

#### Example

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

#### Overview


The PyRelationAL package offers a flexible workflow to enable active learning with as little change to the models and datasets as possible. It is partially inspired by Robert (Munro) Monarch's book: "Human-In-The-Loop Machine Learning" and shares some vocabulary from there. It is principally designed with PyTorch in mind, but can be easily extended to work with other libraries.

For a primer on active learning, we refer the reader to Burr Settles's survey [[reference](https://burrsettles.com/pub/settles.activelearning.pdf)]. In his own words
> The key idea behind active learning is that a machine learning algorithm can
achieve greater accuracy with fewer training labels if it is allowed to choose the
data from which it learns. An active learner may pose queries, usually in the form
of unlabeled data instances to be labeled by an oracle (e.g., a human annotator).
Active learning is well-motivated in many modern machine learning problems,
where unlabeled data may be abundant or easily obtained, but labels are difficult,
time-consuming, or expensive to obtain.

![Overview](docs/images/active_learning_loop.png "Overview")

The `PyRelationAL` package decomposes the active learning workflow into four main components: 1) a **data manager**, 2) a **model**, 3) an **acquisition strategy** built around informativeness scorer, and 4) an **oracle** (see Figure above). Note that the oracle is external to the package.

The data manager (defined in `pyrelational.data.data_manager.GenericDataManager`) wraps around a PyTorch Dataset and handles dataloader instantiation as well as tracking and updating of labelled and unlabelled sample pools.

The model (subclassed from `pyrelational.models.generic_model.GenericModel`) wraps a user defined ML model (e.g. PyTorch Module, Pytorch Lightning Module, or scikit-learn estimator) and handles instantiation, training, testing, as well as uncertainty quantification (e.g. ensembling, MC-dropout). It also enables using ML models that directly estimate their uncertainties such as Gaussian Processes (see `examples/demo/model_gaussianprocesses.py`).

The active learning strategy (which subclass `pyrelational.strategies.generic_al_strategy.GenericActiveLearningStrategy`) revolves around an informativeness score that serve as the basis for the selection of the query sent to the oracle for labelling. We define various strategies for classification, regression, and task-agnostic scenarios based on informativeness scorer defined in `pyrelational.informativeness`.

## Prerequisites and setup

For those just using the package, installation only requires standard ML packages and PyTorch. Starting with a new virtual environment (miniconda environment recommended), install standard learning packages and numerical tools.

```bash
pip install -r requirements.txt
```

If you wish to contribute to the code, run `pre-commit install` after the above step.

## Building the docs

Make sure you have `sphinx` and `sphinx-rtd-theme` packages installed (`pip install sphinx sphinx_rtd_theme` will install this).

To generate the docs, `cd` into the `docs/` directory and run `make html`. This will generate the docs
at `docs/_build/html/index.html`.


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

- examples custom acquisition strategy
  - `model_badge.py`
  - `lightning_mixed_regression.py`

- examples custom model
  - `model_gaussianprocesses.py`

## Uncertainty Estimation

- MCDropout
- Ensemble of models (a.k.a. commitee)
- DropConnect (coming soon)
- SWAG (coming soon)
- MultiSWAG (coming soon)

## Informativeness scorer included in the library

### Regression (N.B. PyRelationAL currently only supports single scalar regression tasks)

- Greedy
- Least confidence
- Expected improvement
- Thompson Sampling
- Upper confidence bound (UCB)
- BALD
- BatchBALD (coming soon)

### Classification (N.B. PyRelationAL does not support multi-label classification at the moment)

- Least confidence
- Margin confidence
- Entropy based confidence
- Ratio based confidence
- BALD
- Thompson Sampling (coming soon)
- BatchBALD (coming soon)


### Model agnostic and diversity sampling based approaches

- Representative sampling
- Diversity sampling
- Random acquisition
- BADGE
