import os
from typing import Any, Dict, Generic, List, Optional, Type, TypeVar, Union, cast

import numpy as np
import torch
from numpy.typing import NDArray

# Ray Tune
from ray import tune
from ray.train import RunConfig
from ray_functions import set_all_seeds
from regression_experiment_utils import (
    GPR,
    experiment_param_space,
    get_strategy_from_string,
    numpy_collate,
)

# Scikit learn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import auc, balanced_accuracy_score, roc_auc_score

from pyrelational.data_managers import DataManager

# Data and data manager
from pyrelational.datasets.regression.uci import UCIAirfoil

# Model, strategy, oracle, and pipeline
from pyrelational.oracles import BenchmarkOracle
from pyrelational.pipeline import Pipeline


# Step 1: Define the get_datamanager function
def get_airfoil_data_manager() -> DataManager:
    ds = UCIAirfoil()
    train_ds, valid_ds, test_ds = torch.utils.data.random_split(ds, [1000, 100, 402])
    train_indices = list(train_ds.indices)
    valid_indices = list(valid_ds.indices)
    test_indices = list(test_ds.indices)

    return DataManager(
        ds,
        train_indices=train_indices,
        validation_indices=valid_indices,
        test_indices=test_indices,
        labelled_indices=np.random.choice(train_indices, 100, replace=False).tolist(),
        loader_batch_size="full",
        loader_collate_fn=numpy_collate,
    )


# Step 2: Define the trial function
def trial(config: Dict[str, Any]) -> Dict[str, Union[float, NDArray[Union[Any, np.float32, np.float64]]]]:
    seed = config["seed"]
    set_all_seeds(seed)
    strategy = get_strategy_from_string(config["strategy"])
    data_manager = get_airfoil_data_manager()
    model_config: Dict[str, Any] = {}
    trainer_config: Dict[str, Any] = {}
    model_manager: GPR = GPR(model_config, trainer_config)
    oracle = BenchmarkOracle()
    pipeline = Pipeline(data_manager=data_manager, model_manager=model_manager, strategy=strategy, oracle=oracle)

    # Annotating data step by step until the trainset is fully annotated
    pipeline.run(num_annotate=1)
    print(pipeline)

    iteration_metrics = []
    for i in range(len(pipeline.performances)):
        if "test_metric" in pipeline.performances[i]:
            iteration_metrics.append(pipeline.performances[i]["test_metric"])

    iteration_metrics = np.array(iteration_metrics)
    score_area_under_curve = auc(np.arange(len(iteration_metrics)), iteration_metrics)

    return {"score": score_area_under_curve, "iteration_metrics": iteration_metrics}


# Step 3: Configure and specift the tuner which will run the trials
experiment_name = "airfoil"
storage_path = os.path.join(os.getcwd(), "benchmark_results")

trial = tune.with_resources(trial, {"cpu": 3})
tuner = tune.Tuner(
    trial,
    tune_config=tune.TuneConfig(num_samples=1),
    param_space=experiment_param_space,
    run_config=RunConfig(
        name=experiment_name,
        storage_path=storage_path,
    ),
)
results = tuner.fit()
