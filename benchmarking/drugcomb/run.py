"""
This script runs the benchmark for the drug combination synergy regression active learning task.
"""

import json
import os
from typing import Any, Dict

import pandas as pd
from tqdm import tqdm

from benchmarking.benchmarking_utils import config_to_string, make_reproducible
from pyrelational.data_managers import DataManager
from pyrelational.datasets import DrugCombDataset
from pyrelational.pipeline import Pipeline
from pyrelational.strategies.regression import (
    LeastConfidenceStrategy,
    ThompsonSamplingStrategy,
    UpperConfidenceBoundStrategy,
)
from pyrelational.strategies.task_agnostic import RandomAcquisitionStrategy

from .data_manager import DrugCombDataManager
from .recover.model_manager import RecoverModelManager

# Setup results folder
results_folder = os.path.join("results")
os.makedirs(results_folder, exist_ok=True)

"""
##################################################
###### Active Learning Experiment Settings
##################################################
"""

# The model and trainer configs should stay same
model_config = {
    "drugs_dim": 1024,
    "cell_lines_dim": 100,
    "encoder_layer_dims": [1024, 128],
    "decoder_layer_dims": [64],
}
trainer_config = {
    "accelerator": "gpu",
    "devices": 1,
    "epochs": 100,
}

experiment_config: Dict[str, Any] = {
    "num_annotate": 256,
    "trainer_config": trainer_config,
    "model_config": model_config,
    "batch_size": 256,
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

    experiment_config["strategy"] = strategy_name
    results_fh = os.path.join(results_folder, config_to_string(experiment_config))
    # Results logging setup
    if os.path.exists(results_fh):
        print(f"Skipping: {results_fh} as it exists.")
        continue
    else:
        os.makedirs(results_fh, exist_ok=False)

    results_dfs = []
    for repetition in tqdm(range(3), desc="Repeated Experiment", leave=False):
        # Pipeline setup
        make_reproducible(repetition)
        data_manager = DrugCombDataManager(seed=repetition, batch_size=experiment_config["batch_size"])
        model_manager = RecoverModelManager(trainer_config=trainer_config, model_config=model_config)
        strategy = strategy_name()

        pipeline = Pipeline(data_manager=data_manager, model_manager=model_manager, strategy=strategy)
        pipeline.run(num_annotate=experiment_config["num_annotate"])

        # Save results
        results_df = pipeline.summary()
        results_df["repetition"] = repetition
        results_dfs.append(results_df)
    pd.concat(results_dfs).to_csv(results_fh + "/results.csv")
    json.dump(experiment_config, open(os.path.join(results_fh, "config.json"), "w"))
