from abc import ABC
from typing import Any, Dict, Generic, List, Optional, Type, TypeVar, Union, cast

import numpy as np
import torch
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.model_helpers import is_overridden
from torch.nn import Module
from torch.utils.data import DataLoader

from .abstract_model_manager import ModelManager
from .lightning_model import LightningModel
from .model_utils import _determine_device

ModelType = TypeVar("ModelType", bound=Module)


class EnsembleManager(Generic[ModelType], ModelManager[ModelType, List[ModelType]], ABC):
    """
    Generic wrapper for ensemble uncertainty estimator
    """

    def __init__(
        self,
        model_class: Type[ModelType],
        model_config: Union[str, Dict[str, Any]],
        trainer_config: Union[str, Dict[str, Any]],
        n_estimators: int = 10,
    ):
        super(EnsembleManager, self).__init__(model_class, model_config, trainer_config)
        self.device = _determine_device(self.trainer_config.get("gpus", 0))
        self.n_estimators = n_estimators

    def __call__(self, loader: DataLoader[Any]) -> torch.Tensor:
        """

        :param loader: pytorch dataloader
        :return: model predictions
        """
        if not self.is_trained():
            raise ValueError("No current model, call 'train(train_loader, valid_loader)' to train the model first")
        models = cast(List[ModelType], self._current_model)
        with torch.no_grad():
            predictions = []
            for model in models:
                model = model.to(self.device)
                model.eval()
                model_prediction = []
                for x, _ in loader:
                    x = x.to(self.device)
                    model_prediction.append(model(x).detach().cpu())
                predictions.append(torch.cat(model_prediction, 0))
            ret = torch.stack(predictions)
        return ret


class LightningEnsembleModel(EnsembleManager[LightningModule], LightningModel):
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
        model_config: Union[Dict[str, Any], str],
        trainer_config: Union[Dict[str, Any], str],
        n_estimators: int = 10,
    ):
        super(LightningEnsembleModel, self).__init__(
            model_class, model_config, trainer_config, n_estimators=n_estimators
        )

    def train(self, train_loader: DataLoader[Any], valid_loader: Optional[DataLoader[Any]] = None) -> None:
        self._current_model = []
        for _ in range(self.n_estimators):
            model = self._init_model()
            trainer, ckpt_callback = self.init_trainer()
            trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=valid_loader)
            if valid_loader is not None and is_overridden("validation_step", model):
                model.load_state_dict(torch.load(ckpt_callback.best_model_path)["state_dict"])
            self._current_model.append(model.cpu())

    def test(self, loader: DataLoader[Any]) -> Dict[str, float]:
        if not self.is_trained():
            raise ValueError("No current model, call 'train(train_loader, valid_loader)' to train the model first")
        trainer, _ = self.init_trainer()
        models = cast(List[LightningModule], self._current_model)
        output = [trainer.test(model, dataloaders=loader)[0] for model in models]
        # return average score across ensemble
        performances: Dict[str, float] = {}
        for k in output[0].keys():
            performances[k] = np.mean([o[k] for o in output]).item()
        return performances
