import os
from typing import Any, Dict, Union

import numpy as np
from numpy.typing import NDArray
from ray import tune
from ray.train import RunConfig
from sklearn.metrics import auc

from pyrelational.oracles import BenchmarkOracle
from pyrelational.pipeline import Pipeline

from ..benchmarking_utils import (  # type: ignore[misc]
    process_results_grid,
    save_results_df,
    set_all_seeds,
)
from ..classification_experiment_utils import (  # type: ignore[misc]
    experiment_param_space,
    get_strategy_from_string,
)
from .data_manager import get_mnist_datamanager
from .model import MCConvNet

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def trial(config: Dict[str, Any]) -> Dict[str, Union[float, NDArray[Union[Any, np.float32, np.float64]]]]:
    seed = config["seed"]
    set_all_seeds(seed)
    strategy = get_strategy_from_string(config["strategy"])
    data_manager = get_mnist_datamanager(data_dir=SCRIPT_DIR, random_state=seed)
    model_config: Dict[str, Any] = {"num_classes": 10}
    trainer_config: Dict[str, Any] = {
        "patience": 3,
        "max_epochs": 100,
        "accelerator": "gpu",
    }
    model_manager = MCConvNet(model_config=model_config, trainer_config=trainer_config)
    oracle = BenchmarkOracle()
    pipeline = Pipeline(data_manager=data_manager, model_manager=model_manager, strategy=strategy, oracle=oracle)

    # Annotating data step by step until the unlabelled set is fully annotated
    pipeline.run(num_annotate=10, num_iterations=1000)
    print(pipeline)

    iteration_metrics = []
    for i in range(len(pipeline.performances)):
        if "test_metric" in pipeline.performances[i]:
            iteration_metrics.append(pipeline.performances[i]["test_metric"])

    iteration_metrics = np.array(iteration_metrics)
    score_area_under_curve = auc(np.arange(len(iteration_metrics)), iteration_metrics)

    return {"score": score_area_under_curve, "iteration_metrics": iteration_metrics}


if __name__ == "__main__":
    EXPERIMENT_NAME = "results"
    STORAGE_PATH = os.path.join(os.getcwd(), "ray_benchmark_results")

    # Download data once and for all
    _ = get_mnist_datamanager(data_dir=SCRIPT_DIR)

    trial = tune.with_resources(trial, {"cpu": 4, "gpu": 0.25})
    tuner = tune.Tuner(
        trial,
        tune_config=tune.TuneConfig(num_samples=1),
        param_space=experiment_param_space,
        run_config=RunConfig(
            name=EXPERIMENT_NAME,
            storage_path=STORAGE_PATH,
        ),
    )
    results_grid = tuner.fit()
    results_df = process_results_grid(results_grid=results_grid)
    save_results_df(results_df=results_df, storage_path=SCRIPT_DIR, experiment_name=EXPERIMENT_NAME)
