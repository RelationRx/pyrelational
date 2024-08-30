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
from pyrelational.strategies.classification import LeastConfidenceStrategy, EntropyClassificationStrategy, MarginalConfidenceStrategy, RatioConfidenceStrategy 
from pyrelational.strategies.task_agnostic import RandomAcquisitionStrategy

# Ray Tune
from ray import tune
from ray.train import RunConfig
import ray
import os

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

# Wrapping the RFC with pyrelational's ModelManager
class SKRFC(ModelManager):
    """
    Scikit learn RandomForestClassifier implementing the interface of our ModelManager
    for active learning.
    """

    def __init__(self, model_class, model_config, trainer_config):
        super(SKRFC, self).__init__(model_class, model_config, trainer_config)

    def train(self, train_loader, valid_loader):
        train_x, train_y = next(iter(train_loader))
        estimator = self._init_model()
        estimator.fit(train_x, train_y)
        self._current_model = estimator

    def test(self, loader):
        if not self.is_trained():
            raise ValueError("No current model, call 'train(X, y)' to train the model first")
        X, y = next(iter(loader))
        y_hat = self._current_model.predict(X)
        metric = balanced_accuracy_score(y_hat, y)
        return {"test_metric": metric}

    def __call__(self, loader):
        if not self.is_trained():
            raise ValueError("No current model, call 'train(X, y)' to train the model first")
        X, _ = next(iter(loader))
        model = self._current_model
        class_probabilities = model.predict_proba(X)
        return torch.FloatTensor(class_probabilities).unsqueeze(0)  # unsqueeze due to batch expectation

def get_strategy_from_string(strategy):
    if strategy == "least_confidence":
        return LeastConfidenceStrategy()
    elif strategy == "entropy":
        return EntropyClassificationStrategy()
    elif strategy == "marginal_confidence":
        return MarginalConfidenceStrategy()
    elif strategy == "ratio_confidence":
        return RatioConfidenceStrategy()
    else:
        raise ValueError("Invalid strategy")

experiment_param_space = {
    "seed": tune.grid_search([1,2,3,4,5]),
    "strategy": tune.grid_search(["least_confidence", "entropy", "marginal_confidence", "ratio_confidence"])
}

# config = {
#     "seed": 1,
#     "strategy": "least_confidence"
# }

def trial(config):
    seed = config["seed"]
    strategy = get_strategy_from_string(config["strategy"])
    data_manager = get_breastcancer_data_manager()
    model_config = {"n_estimators": 10, "bootstrap": False}
    trainer_config = {}
    model_manager = SKRFC(RandomForestClassifier, model_config, trainer_config)

    # Instantiate an active learning strategy
    al_strategy = RandomAcquisitionStrategy()

    # Instantiate an oracle
    oracle = BenchmarkOracle()

    # Given that we have a data manager, a model_manager, and an active learning strategy
    # we may create an active learning pipeline
    pipeline = Pipeline(data_manager=data_manager, model_manager=model_manager, strategy=al_strategy, oracle=oracle)

    # Annotating data step by step until the trainset is fully annotated
    pipeline.run(num_annotate=1)
    print(pipeline)

    iteration_metrics = []
    for i in range(len(pipeline.performances)):
        if "test_metric" in pipeline.performances[i]:
            iteration_metrics.append(pipeline.performances[i]["test_metric"])

    print(iteration_metrics)
    iteration_metrics = np.array(iteration_metrics)
    score_area_under_curve = auc(np.arange(len(iteration_metrics)), iteration_metrics)
        
    return {"score": score_area_under_curve, "iteration_metrics": iteration_metrics}

# Configure and specift the tuner which will run the trials
experiment_name = "breastcancer"
storage_path = os.path.join(os.getcwd(), experiment_name + "_results")

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