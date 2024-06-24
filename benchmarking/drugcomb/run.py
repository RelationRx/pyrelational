"""
This script runs the benchmark for the drug combination synergy regression active learning task.
"""

import json
import os
import pathlib
from typing import Any, Dict

import pandas as pd
from tqdm import tqdm

from benchmarking.benchmarking_utils import hash_dictionary, make_reproducible
from benchmarking.drugcomb.data_manager import DrugCombDataManager
from benchmarking.drugcomb.recover.model_manager import RecoverModelManager
from pyrelational.pipeline import Pipeline
from pyrelational.strategies.regression import (
    LeastConfidenceStrategy,
    ThompsonSamplingStrategy,
    UpperConfidenceBoundStrategy,
)
from pyrelational.strategies.task_agnostic import RandomAcquisitionStrategy

# Setup results folder
results_folder = os.path.join(pathlib.Path(__file__).parent.resolve(), "results")
os.makedirs(results_folder, exist_ok=True)

"""
##################################################
###### Active Learning Experiment Settings
##################################################
"""

# The model and trainer configs should stay same
model_config = {
    "drugs_dim": 1024,
    "cell_lines_dim": 512,
    "encoder_layer_dims": [1024, 128],
    "decoder_layer_dims": [64, 1],
}
trainer_config = {
    "accelerator": "gpu",
    "devices": 1,
    "epochs": 100,
    "monitor_metric_name": "mse",
}

experiment_config: Dict[str, Any] = {
    "num_annotate": 1024,
    "trainer_config": trainer_config,
    "model_config": model_config,
    "batch_size": 2048,
    "num_workers": 8,
}

"""
##################################################
###### Benchmarking settings
##################################################
"""
strategies = [
    LeastConfidenceStrategy,
    RandomAcquisitionStrategy,
    UpperConfidenceBoundStrategy,
    ThompsonSamplingStrategy,
]

for strategy_name in tqdm(strategies, desc="Strategy Progress"):

    experiment_config["strategy"] = strategy_name.__class__.__name__
    results_fh = os.path.join(results_folder, hash_dictionary(experiment_config))
    # Results logging setup
    if os.path.exists(results_fh):
        print(f"Skipping: {results_fh} as it exists.")
        continue

    results_dfs = []
    for repetition in tqdm(range(3), desc="Repeated Experiment", leave=False):
        # Pipeline setup
        make_reproducible(repetition)
        data_manager = DrugCombDataManager(
            seed=repetition, batch_size=experiment_config["batch_size"], num_workers=experiment_config["num_workers"]
        )
        model_manager = RecoverModelManager(trainer_config=trainer_config, model_config=model_config)
        strategy = strategy_name()

        pipeline = Pipeline(data_manager=data_manager, model_manager=model_manager, strategy=strategy)
        pipeline.run(num_annotate=experiment_config["num_annotate"])

        # Save results
        results_df = pipeline.summary()
        results_df["repetition"] = repetition
        results_dfs.append(results_df)
    os.makedirs(results_fh, exist_ok=False)
    pd.concat(results_dfs).to_csv(os.path.join(results_fh, "results.csv"))
    json.dump(experiment_config, open(os.path.join(results_fh, "config.json"), "w"))
