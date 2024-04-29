"""
This script runs the benchmark for the Synthetic 2D regression active learning task
"""

import json
import os
from typing import Any, Dict

import pandas as pd

# from regression_eval_pipeline import RegressionEvalPipeline
from data_manager import Synthetic2D, Synthetic2DDataManager
from ensemble_scikit import EnsembleScikit
from sklearn.neural_network import MLPRegressor
from tqdm import tqdm

from benchmarking.benchmarking_utils import config_to_string
from pyrelational.oracles import BenchmarkOracle
from pyrelational.pipeline import Pipeline
from pyrelational.strategies.regression import (
    BALDStrategy,
    LeastConfidenceStrategy,
    ThompsonSamplingStrategy,
    UpperConfidenceBoundStrategy,
)
from pyrelational.strategies.task_agnostic import RandomAcquisitionStrategy

# Setup results folder
results_folder = os.path.join("results", "single_acquisition")
os.makedirs(results_folder, exist_ok=True)

"""
##################################################
###### Active Learning Experiment Settings
##################################################
"""

# The model and trainer configs should stay same
model_config = {
    "hidden_layer_sizes": (
        64,
        64,
        32,
    ),
    "activation": "relu",
    "max_iter": 1000,
    "n_iter_no_change": 10,
    "tol": 1e-3,
    "batch_size": 32,
}
trainer_config: Dict[str, Any] = {}  # scikit learn has train configs inside model_config
ensemble_size = 10

num_annotate = 1
num_iterations = 10

experiment_config = {}

"""
##################################################
###### Benchmarking settings
##################################################
"""
strategies = [LeastConfidenceStrategy, RandomAcquisitionStrategy]

for strategy_name in tqdm(strategies, desc="Strategy Progress"):

    experiment_config["strategy"] = strategy_name
    results_fh = os.path.join(results_folder, config_to_string(experiment_config))
    # Results logging setup
    if os.path.exists(results_fh):
        print(f"Skipping: {results_fh} as it exists.")
        continue
    else:
        os.makedirs(results_fh, exist_ok=False)

    results_dfs = []
    for repetition in tqdm(range(2), desc="Repeated Experiment", leave=False):

        # Pipeline setup
        dataset = Synthetic2D()
        data_manager = Synthetic2DDataManager(dataset=dataset, random_state=42, numpy_flag=True)

        model_config["random_state"] = repetition
        model_manager = EnsembleScikit(
            model_class=MLPRegressor,
            num_estimators=ensemble_size,
            model_config=model_config,
            trainer_config=trainer_config,
        )
        oracle = BenchmarkOracle()
        strategy = strategy_name()

        pipeline = Pipeline(data_manager=data_manager, model_manager=model_manager, strategy=strategy, oracle=oracle)
        pipeline.run(num_annotate=num_annotate, num_iterations=num_iterations)

        # Save results
        results_df = pipeline.summary()
        results_df["repetition"] = repetition
        results_dfs.append(results_df)
    pd.concat(results_dfs).to_csv(results_fh + "/results.csv")
    json.dump(experiment_config, open(os.path.join(results_fh, "config.json"), "w"))
