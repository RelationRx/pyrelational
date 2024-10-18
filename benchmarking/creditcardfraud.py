import os
from typing import Any, Dict, Generic, List, Optional, Type, TypeVar, Union, cast

import numpy as np
import torch
from benchmarking_utils import process_results_grid, save_results_df, set_all_seeds
from classification_experiment_utils import (
    SKRFC,
    experiment_param_space,
    get_strategy_from_string,
    numpy_collate,
)
from numpy.typing import NDArray
from ray import tune
from ray.train import RunConfig
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import auc, balanced_accuracy_score, roc_auc_score

from pyrelational.data_managers import DataManager
from pyrelational.datasets.classification.andrea_et_al import CreditCardDataset
from pyrelational.oracles import BenchmarkOracle
from pyrelational.pipeline import Pipeline


def get_creditcard_data_manager() -> DataManager:
    ds = CreditCardDataset()
    train_ds, valid_ds, test_ds = torch.utils.data.random_split(ds, [142407, 35600, 106800])
    train_indices = list(train_ds.indices)
    valid_indices = list(valid_ds.indices)
    test_indices = list(test_ds.indices)

    return DataManager(
        ds,
        train_indices=train_indices,
        validation_indices=valid_indices,
        test_indices=test_indices,
        labelled_indices=np.random.choice(train_indices, 2500, replace=False).tolist(),
        loader_batch_size="full",
        loader_collate_fn=numpy_collate,
    )


def trial(config: Dict[str, Any]) -> Dict[str, Union[float, NDArray[Union[Any, np.float32, np.float64]]]]:
    seed = config["seed"]
    set_all_seeds(seed)
    strategy = get_strategy_from_string(config["strategy"])
    data_manager = get_creditcard_data_manager()
    model_config = {"n_estimators": 10, "bootstrap": False}
    trainer_config: Dict[str, Any] = {}
    model_manager = SKRFC(RandomForestClassifier, model_config, trainer_config)
    oracle = BenchmarkOracle()
    pipeline = Pipeline(data_manager=data_manager, model_manager=model_manager, strategy=strategy, oracle=oracle)

    # Annotating data step by step until the trainset is fully annotated
    pipeline.run(num_annotate=1, num_iterations=1000)
    print(pipeline)

    iteration_metrics = []
    for i in range(len(pipeline.performances)):
        if "test_metric" in pipeline.performances[i]:
            iteration_metrics.append(pipeline.performances[i]["test_metric"])

    iteration_metrics = np.array(iteration_metrics)
    score_area_under_curve = auc(np.arange(len(iteration_metrics)), iteration_metrics)

    return {"score": score_area_under_curve, "iteration_metrics": iteration_metrics}


# Configure and specift the tuner which will run the trials
experiment_name = "creditcardfraud"
storage_path = os.path.join(os.getcwd(), "ray_benchmark_results")

trial = tune.with_resources(trial, {"cpu": 4})
tuner = tune.Tuner(
    trial,
    tune_config=tune.TuneConfig(num_samples=1),
    param_space=experiment_param_space,
    run_config=RunConfig(
        name=experiment_name,
        storage_path=storage_path,
    ),
)
results_grid = tuner.fit()
results_df = process_results_grid(results_grid=results_grid)
save_results_df(results_df=results_df, storage_path="benchmark_results", experiment_name=experiment_name)


# ######## Local test ########
# from tqdm import tqdm
# configs = []
# strategies = ["random", "least_confidence", "entropy", "marginal_confidence", "ratio_confidence"]
# for seed in [1]:
#     print(f"Running config")
#     for strategy in strategies:
#         configs.append({"seed": seed, "strategy": strategy})
# results = []
# for config in tqdm(configs, desc="Running trials"):
#     print(f"Running config: {config}")
#     results.append(trial(config))
# print(results)
