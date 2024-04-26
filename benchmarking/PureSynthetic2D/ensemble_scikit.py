from abc import ABC
from typing import Any, Dict, Generic, List, Optional, Type, TypeVar, Union, cast

import numpy as np
import sklearn
import torch
from sklearn.metrics import explained_variance_score, mean_squared_error, r2_score
from sklearn.model_selection import KFold

from pyrelational.model_managers import ModelManager


class EnsembleScikit(ModelManager[sklearn.base.BaseEstimator, List[sklearn.base.BaseEstimator]]):
    """Custom ModelManager for an ensemble of K-fold SVM regressors, as found in query-by-commitee
    strategies. Can easily be adapted to any other regressor

    Args:
    model_class (sklearn estimator): Estimator that should be ensembled (e.g. MLPRegressor)
    num_estimators (int): number of estimators in the ensemble
    model_config (dict): dictionary containing any model_class specific arguments
    trainer_config (dict): dictionary containing any taining specific arguments
    """

    def __init__(
        self,
        model_class: sklearn.base.BaseEstimator,
        num_estimators: int,
        model_config: Dict[str, Any],
        trainer_config: Dict[str, Any],
    ):
        super(EnsembleScikit, self).__init__(model_class, model_config, trainer_config)
        self.model_config = model_config
        self.trainer_config = trainer_config
        self.num_estimators = num_estimators

    def train(
        self,
        train_loader: torch.utils.data.DataLoader[Any],
        validation_loader: Optional[torch.utils.data.DataLoader[Any]] = None,
    ) -> None:
        """
        Args:
        train_loader (torch.utils.data.DataLoader): A torch dataloader with a numpy compatible collate function
            you can also just adapt this to something else.
        """
        train_x, train_y = next(iter(train_loader))  # assumes dataloader returns full set of available observations
        estimators: List[Any] = []

        k = self.num_estimators
        kf = KFold(n_splits=k)
        from joblib import Parallel, delayed

        estimators = Parallel(n_jobs=-1)(
            delayed(self._init_model().fit)(train_x[train_index], train_y[train_index])
            for train_index, _ in kf.split(train_x)
        )

        # Set the current model to the list of trained regressors
        self._current_model: List[Any] = estimators

    def test(self, loader: torch.utils.data.DataLoader[Any]) -> Dict[str, float]:
        if not self.is_trained():
            raise ValueError("No current model, call 'train(X, y)' to train the model first")
        X, y = next(iter(loader))
        scores = []
        for idx in range(self.num_estimators):
            estimator = self._current_model[idx]
            predictions = estimator.predict(X)
            score_mse = mean_squared_error(y, predictions)
            score_r2 = r2_score(y, predictions)
            score_explained_variance = explained_variance_score(y, predictions)
            scores.append(score_mse)
            scores.append(score_r2)
            scores.append(score_explained_variance)
        return {
            "MSE": np.mean(score_mse),
            "R2": np.mean(score_r2),
            "Explained Variance": np.mean(score_explained_variance),
        }

    def __call__(self, loader: torch.utils.data.DataLoader[Any]) -> torch.Tensor:
        if not self.is_trained():
            raise ValueError("No current model, call 'train(X, y)' to train the model first")
        X, _ = next(iter(loader))
        predictions: List[torch.Tensor] = []  # list of num_estimator predictions of shape y
        for est_idx in range(self.num_estimators):
            estimator = self._current_model[est_idx]
            predictions.append(torch.FloatTensor(estimator.predict(X)))
        predictions = torch.vstack(predictions)
        return predictions
