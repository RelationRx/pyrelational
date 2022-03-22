from typing import Dict, Tuple, Type, Union

import pytorch_lightning as pl
import torch
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.utilities.model_helpers import is_overridden
from torch.utils.data import DataLoader

from .generic_model import GenericModel


class LightningModel(GenericModel):
    r"""
    A wrapper for pytorch lightning modules that instantiates and uses a pytorch lightning trainer.

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

        wrapper = LightningModel(
                        PyLModel,
                        model_config={"in_dim":10, "out_dim":1},
                        trainer_config={"epochs":100},
                  )
        wrapper.train(train_loader, valid_loader)
    """

    def __init__(
        self,
        model_class: Type[LightningModule],
        model_config: Union[Dict, str],
        trainer_config: Union[Dict, str],
    ):
        super(LightningModel, self).__init__(model_class, model_config, trainer_config)

    def init_trainer(self) -> Tuple[Trainer, ModelCheckpoint]:
        """

        :param config: dictionary of key:value pairs required to instantiate a trainer object
        :return: a pytorch lightning trainer object
        """
        config = self.trainer_config
        config = _check_pyl_trainer_config(config)
        callbacks = []
        if config["use_early_stopping"]:
            callbacks.append(
                EarlyStopping(
                    monitor=config["monitor_metric_name"],
                    patience=config["patience"],
                    verbose=True,
                    mode=config["monitor_metric_mode"],
                )
            )
        checkpoint_callback = ModelCheckpoint(
            monitor=config["monitor_metric_name"],
            dirpath=config["checkpoints_dir"],
            filename=config["checkpoints_name"],
            save_top_k=config["save_top_k"],
            mode=config["monitor_metric_mode"],
        )
        callbacks.append(checkpoint_callback)

        tracker = pl.loggers.TensorBoardLogger(save_dir=config["checkpoints_dir"], name=config["checkpoints_name"])
        trainer = pl.Trainer(
            callbacks=callbacks,
            logger=tracker,
            gpus=config["gpus"],
            max_epochs=config["epochs"],
            check_val_every_n_epoch=config["period_eval"],
            log_every_n_steps=1,
        )
        return trainer, checkpoint_callback

    def train(self, train_loader: DataLoader, valid_loader: DataLoader = None) -> None:
        trainer, ckpt_callback = self.init_trainer()

        model = self.init_model()
        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=valid_loader)
        if valid_loader is not None and is_overridden("validation_step", model):
            model.load_state_dict(torch.load(ckpt_callback.best_model_path)["state_dict"])

        self.current_model = model

    def test(self, loader: DataLoader) -> Dict:
        if self.current_model is None:
            raise ValueError("No current model, call 'train(train_loader, valid_loader)' to train the model first")
        trainer, _ = self.init_trainer()
        return trainer.test(self.current_model, dataloaders=loader)[0]

    def __call__(self, loader: DataLoader) -> torch.Tensor:
        """

        :param loader: pytorch dataloader
        :return: model predictions
        """
        if self.current_model is None:
            raise ValueError("No current model, call 'train(train_loader, valid_loader)' to train the model first")
        model = self.current_model
        model.eval()

        with torch.no_grad():
            model_prediction = []
            for x, _ in loader:
                model_prediction.append(model(x).detach().cpu())
        predictions = torch.cat(model_prediction, 0)
        return predictions


def _check_pyl_trainer_config(config: Dict) -> Dict:
    """
    Checks the trainer config for pytorch lightning and adds default values for missing required entries
    :param config: a dictionary with key:values required by the init_trainer function
    :return: dictionary with trainer config
    """
    default = {
        "gpus": 0,
        "epochs": 100,
        "period_eval": 1,
        "checkpoints_dir": "experiment_logs/",
        "checkpoints_name": "run",
        "monitor_metric_name": "loss",
        "monitor_metric_mode": "min",
        "use_early_stopping": False,
        "patience": 100,
        "save_top_k": 1,
    }
    config = {**default, **config}
    return config
