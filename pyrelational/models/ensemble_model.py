from abc import ABC
from typing import Any, Dict, List, Optional, Type, Union

import numpy as np
import torch
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.model_helpers import is_overridden
from torch.utils.data import DataLoader

from .abstract_model_manager import ModelManager
from .lightning_model import LightningModel
from .model_utils import _determine_device


class EnsembleManager(ModelManager, ABC):
    """
    Generic wrapper for ensemble uncertainty estimator
    """

    def __init__(
        self,
        model_class: Type[Any],
        model_config: Union[str, Dict],
        trainer_config: Union[str, Dict],
        n_estimators: int = 10,
    ):
        """
        :param model_class: a model constructor (e.g. torch.nn.Linear)
        :param model_config: a dictionary containing the config required to instantiate a model form the model_class
                (e.g. {in_features=100, out_features=34, bias=True, device=None, dtype=None} for a torch.nn.Linear
                constructor)
        :param trainer_config: a dictionary containing the config required to instantiate the trainer module/function
        :param n_estimators: number of models in ensemble
        """
        super(EnsembleManager, self).__init__(model_class, model_config, trainer_config)
        self.device = _determine_device(self.trainer_config.get("gpus", 0))
        self.n_estimators = n_estimators

    def init_model(self) -> List[Any]:
        """
        Initialise ensemble, ie a list of model instances

        :return: list of models of length n_estimators
        """
        return [self.model_class(**self.model_config) for _ in range(self.n_estimators)]

    def __call__(self, loader: DataLoader) -> torch.Tensor:
        """
        Call method to output model predictions for each model in the ensemble

        :param loader: pytorch dataloader
        :return: model predictions of shape (n_estimators, number of samples in loader, 1)
        """
        if self.current_model is None:
            raise ValueError("No current model, call 'train(train_loader, valid_loader)' to train the model first")

        with torch.no_grad():
            predictions = []
            for model in self.current_model:
                model = model.to(self.device)
                model.eval()
                model_prediction = []
                for x, _ in loader:
                    x = x.to(self.device)
                    model_prediction.append(model(x).detach().cpu())
                predictions.append(torch.cat(model_prediction, 0))
            predictions = torch.stack(predictions)
        return predictions


class LightningEnsembleModel(EnsembleManager, LightningModel):
    r"""
    Wrapper for ensemble estimator with pytorch lightning trainer


    Example:

    .. code-block:: python

        import torch
        import pytorch_lightning as pl

        class PyLModel(pl.LightningModule):
           def __init__(self, in_dim, out_dim):
               super(PyLModel, self).()
               self.linear = torch.nn.Linear(in_dim, out_dim)
        # need to define other train/test steps and optimizers methods required
        # by pytorch-lightning to run this example

        wrapper = LightningEnsembleModel(
                     PyLModel,
                     model_config={"in_dim":10, "out_dim":1},
                     trainer_config={"epochs":100},
                     n_estimators=10,
               )
        wrapper.train(train_loader, valid_loader)
        predictions = wrapper(loader)
        assert predictions.size(0) == 10
    """

    def __init__(
        self,
        model_class: Type[LightningModule],
        model_config: Union[Dict, str],
        trainer_config: Union[Dict, str],
        n_estimators: int = 10,
    ):
        """
        :param model_class: a model constructor class which inherits from pytorch lightning (see above example)
        :param model_config: a dictionary containing the config required to instantiate a model form the model_class
                (e.g. see above example)
        :param trainer_config: a dictionary containing the config required to instantiate the pytorch lightning trainer
        :param n_estimators: number of models in ensemble
        """
        super(LightningEnsembleModel, self).__init__(
            model_class, model_config, trainer_config, n_estimators=n_estimators
        )

    def train(self, train_loader: DataLoader, valid_loader: Optional[DataLoader] = None) -> None:
        """
        Train all models in ensemble
        """
        models = self.init_model()
        self.current_model = []
        for model in models:
            trainer, ckpt_callback = self.init_trainer()
            trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=valid_loader)
            if valid_loader is not None and is_overridden("validation_step", model):
                model.load_state_dict(torch.load(ckpt_callback.best_model_path)["state_dict"])
            self.current_model.append(model.cpu())

    def test(self, loader: DataLoader) -> Dict:
        """
        Test ensemble model. The mean performance across all the models in the ensemble is reported
        for each metric

        :param loader: dataloader for test set

        :return: average performance for each metric (defined in the model_class)
        """
        if self.current_model is None:
            raise ValueError("No current model, call 'train(train_loader, valid_loader)' to train the model first")
        trainer, _ = self.init_trainer()
        output = [trainer.test(model, dataloaders=loader)[0] for model in self.current_model]
        # return average score across ensemble
        performances = {}
        for k in output[0].keys():
            performances[k] = np.mean([o[k] for o in output])
        return performances
