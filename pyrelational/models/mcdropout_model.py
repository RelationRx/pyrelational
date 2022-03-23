import copy
import logging
from abc import ABC
from typing import Dict, Optional, Type, Union

import torch
from pytorch_lightning import LightningModule
from torch.nn.modules import Module
from torch.utils.data import DataLoader

from .generic_model import GenericModel
from .lightning_model import LightningModel

logger = logging.getLogger()


class GenericMCDropoutModel(GenericModel, ABC):
    """
    Generic model wrapper for mcdropout uncertainty estimator
    """

    def __init__(
        self,
        model_class: Type[Module],
        model_config: Union[str, Dict],
        trainer_config: Union[str, Dict],
        n_estimators: int = 10,
        eval_dropout_prob: float = 0.2,
    ):
        super(GenericMCDropoutModel, self).__init__(model_class, model_config, trainer_config)
        _check_mc_dropout_model(model_class, model_config)
        self.n_estimators = n_estimators
        self.eval_dropout_prob = eval_dropout_prob

    def __call__(self, loader: DataLoader) -> torch.Tensor:
        """

        :param loader: pytorch dataloader
        :return: model predictions
        """
        if self.current_model is None:
            raise ValueError("No current model, call 'train(train_loader, valid_loader)' to train the model first")
        predictions = []
        model = self.current_model
        model.eval()

        with torch.no_grad():
            _enable_only_dropout_layers(model, self.eval_dropout_prob)
            for _ in range(self.n_estimators):
                model_prediction = []
                for x, _ in loader:
                    model_prediction.append(model(x).detach().cpu())
                predictions.append(torch.cat(model_prediction, 0))
            predictions = torch.stack(predictions)
        return predictions


class LightningMCDropoutModel(GenericMCDropoutModel, LightningModel):
    r"""
    Wrapper for MC Dropout estimator with pytorch lightning trainer

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

        wrapper = LightningMCDropoutModel(
                        PyLModel,
                        model_config={"in_dim":10, "out_dim":1},
                        trainer_config={"epochs":100},
                        n_estimators=10,
                        eval_dropout_prob=0.2,
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
        eval_dropout_prob: float = 0.2,
    ):
        super(LightningMCDropoutModel, self).__init__(
            model_class,
            model_config,
            trainer_config,
            n_estimators=n_estimators,
            eval_dropout_prob=eval_dropout_prob,
        )


def _enable_only_dropout_layers(model: Module, p: Optional[float] = None) -> None:
    def enable_dropout_on_module(m):
        if m.__class__.__name__.startswith("Dropout"):
            if isinstance(p, float) and (0 <= p <= 1):
                m.p = p
            elif isinstance(p, float) and (p < 0 or p > 1):
                logger.warning(f"Evaluation dropout probability should be a float between 0 and 1, got {p}")
            m.train()

    model.apply(enable_dropout_on_module)


def _check_mc_dropout_model(model_class: Type[Module], model_config: Dict) -> None:
    model = model_class(**model_config)

    def has_dropout_module(model):
        is_dropout = []
        for m in model.children():
            if m.__class__.__name__.startswith("Dropout"):
                is_dropout.append(True)
            else:
                is_dropout += has_dropout_module(m)
        return is_dropout

    if not any(has_dropout_module(model)):
        raise ValueError("Model provided do not contain any torch.nn.Dropout modules, cannot apply MC Dropout")
