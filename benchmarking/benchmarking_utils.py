import os
import pathlib
import random
from typing import Union

import numpy as np
import pandas as pd
import torch
from ray.tune.result_grid import ResultGrid
from tqdm import tqdm


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def process_results_grid(results_grid: ResultGrid) -> pd.DataFrame:
    # Check if there have been errors
    if results_grid.errors:
        print("One of the trials failed!")
    else:
        print("No errors!")

    num_results = len(results_grid)
    print("Number of results:", num_results)

    # Iterate over results
    for i, result in enumerate(results_grid):  # type: ignore
        if result.error:
            print(f"Trial #{i} had an error:", result.error)
            continue

        print(f"Trial #{i} finished successfully with a test metric of:", result.metrics["score"])

    # Process the results grid to obtain a DataFrame that looks at the
    results_df = results_grid.get_dataframe()
    results_df = process_results_grid_into_sns_plot_df(results_df=results_df)

    return results_df


def process_results_grid_into_sns_plot_df(results_df: pd.DataFrame) -> pd.DataFrame:
    """Process the results grid into a data frame so that we can plot line plots with seaborn.

    The current results_df format has each row as a trial, with columns for the trial's parameters and metrics.
    We want to create a data frame that has a row for each trial and each element in the list of iteration_metrics,
    such that we can plot the metrics over the iterations. The resulting data frame will have the following columns:

    - trial_id: The trial's ID
    - iteration: The iteration number
    - test_metric: The test metric at that iteration
    - strategy: The strategy used in that iteration
    - seed: The seed used in that trial
    - score: The score for that trial
    """
    sns_plot_df = []
    for _, row in tqdm(results_df.iterrows()):
        trial_id = row["trial_id"]
        seed = row["config/seed"]
        score = row["score"]

        for iteration, metric_value in enumerate(row["iteration_metrics"]):
            sns_plot_df.append(
                {
                    "trial_id": trial_id,
                    "iteration": iteration,
                    "test_metric": metric_value,
                    "strategy": row["config/strategy"],
                    "seed": seed,
                    "score": score,
                }
            )

    return pd.DataFrame(sns_plot_df)


def save_results_df(results_df: pd.DataFrame, storage_path: Union[str, pathlib.Path], experiment_name: str) -> None:
    os.makedirs(storage_path, exist_ok=True)
    results_df.to_csv(os.path.join(storage_path, f"{experiment_name}.csv"))
