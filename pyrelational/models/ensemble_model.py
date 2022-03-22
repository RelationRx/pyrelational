from abc import ABC
from typing import Any, Dict, List, Optional, Type, Union

import numpy as np
import torch
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.model_helpers import is_overridden
from torch.utils.data import DataLoader

from .generic_model import GenericModel
from .lightning_model import LightningModel


class GenericEnsembleModel(GenericModel, ABC):
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
        super(GenericEnsembleModel, self).__init__(model_class, model_config, trainer_config)
        self.n_estimators = n_estimators

    def init_model(self) -> List[Any]:
        """

        :return: list of models
        """
        return [self.model_class(**self.model_config) for _ in range(self.n_estimators)]

    def __call__(self, loader: DataLoader) -> torch.Tensor:
        """

        :param loader: pytorch dataloader
        :return: model predictions
        """
        if self.current_model is None:
            raise ValueError("No current model, call 'train(train_loader, valid_loader)' to train the model first")

        with torch.no_grad():
            predictions = []
            for model in self.current_model:
                model.eval()
                model_prediction = []
                for x, _ in loader:
                    model_prediction.append(model(x).detach().cpu())
                predictions.append(torch.cat(model_prediction, 0))
            predictions = torch.stack(predictions)
        return predictions


class LightningEnsembleModel(GenericEnsembleModel, LightningModel):
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
        super(LightningEnsembleModel, self).__init__(
            model_class, model_config, trainer_config, n_estimators=n_estimators
        )

    def train(self, train_loader: DataLoader, valid_loader: Optional[DataLoader] = None) -> None:
        models = self.init_model()
        self.current_model = []
        for model in models:
            trainer, ckpt_callback = self.init_trainer()
            trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=valid_loader)
            if valid_loader is not None and is_overridden("validation_step", model):
                model.load_state_dict(torch.load(ckpt_callback.best_model_path)["state_dict"])
            self.current_model.append(model.cpu())

    def test(self, loader: DataLoader) -> Dict:
        if self.current_model is None:
            raise ValueError("No current model, call 'train(train_loader, valid_loader)' to train the model first")
        trainer, _ = self.init_trainer()
        output = [trainer.test(model, dataloaders=loader)[0] for model in self.current_model]
        # return average score across ensemble
        performances = {}
        for k in output[0].keys():
            performances[k] = np.mean([o[k] for o in output])
        return performances
