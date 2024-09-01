import numpy as np
import torch

# Scikit learn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, auc, roc_auc_score

# Data and data manager
from pyrelational.datasets.classification.scikit_learn import BreastCancerDataset
from pyrelational.data_managers import DataManager

# Model, strategy, oracle, and pipeline
from pyrelational.model_managers import ModelManager
from pyrelational.oracles import BenchmarkOracle
from pyrelational.pipeline import Pipeline

# Ray Tune
from ray import tune
from ray.train import RunConfig
import os

from classification_experiment_utils import get_strategy_from_string
from classification_experiment_utils import SKRFC
from classification_experiment_utils import experiment_param_space

def numpy_collate(batch):
    """Collate function for a Pytorch to Numpy DataLoader"""
    return [np.stack(el) for el in zip(*batch)]

def get_breastcancer_data_manager():
    ds = BreastCancerDataset()
    train_ds, valid_ds, test_ds = torch.utils.data.random_split(ds, [300, 100, 169])
    train_indices = train_ds.indices
    valid_indices = valid_ds.indices
    test_indices = test_ds.indices

    return DataManager(
        ds,
        train_indices=train_indices,
        validation_indices=valid_indices,
        test_indices=test_indices,
        labelled_indices=np.random.choice(train_indices, 10, replace=False).tolist(),
        loader_batch_size="full",
        loader_collate_fn=numpy_collate,
    )

def trial(config):
    seed = config["seed"]
    strategy = get_strategy_from_string(config["strategy"])
    data_manager = get_breastcancer_data_manager()
    model_config = {"n_estimators": 10, "bootstrap": False}
    trainer_config = {}
    model_manager = SKRFC(RandomForestClassifier, model_config, trainer_config)
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

# Configure and specift the tuner which will run the trials
experiment_name = "breastcancer"
storage_path = os.path.join(os.getcwd(), "benchmark_results")

trial = tune.with_resources(trial, {"cpu": 3})
tuner = tune.Tuner(
    trial,
    tune_config=tune.TuneConfig(num_samples=1),
    param_space=experiment_param_space,
    run_config = RunConfig(
        name = experiment_name,
        storage_path = storage_path,)
)
results = tuner.fit()