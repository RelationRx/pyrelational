from typing import Any, Dict, List, Optional, Tuple, Type, Union, cast

import pytorch_lightning as pl
import torch
from pytorch_lightning import Callback, LightningModule, Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.utilities.model_helpers import is_overridden
from torch.utils.data import DataLoader

from .abstract_model_manager import ModelManager
from .model_utils import _determine_device


class LightningModelManager(ModelManager[LightningModule, LightningModule]):
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

        wrapper = LightningModelManager(
                        PyLModel,
                        model_config={"in_dim":10, "out_dim":1},
                        trainer_config={"epochs":100},
                  )
        wrapper.train(train_loader, valid_loader)
    """

    def __init__(
        self,
        model_class: Type[LightningModule],
        model_config: Union[Dict[str, Any], str],
        trainer_config: Union[Dict[str, Any], str],
    ):
        """
        :param model_class: a model constructor class which inherits from pytorch lightning (see above example)
        :param model_config: a dictionary containing the config required to instantiate a model form the model_class
                (e.g. see above example)
        :param trainer_config: a dictionary containing the config required to instantiate the pytorch lightning trainer
        """
        super(LightningModelManager, self).__init__(model_class, model_config, trainer_config)
        self.device = _determine_device(self.trainer_config.get("gpus", 0))

    def init_trainer(self) -> Tuple[Trainer, ModelCheckpoint]:
        """
        Initialise pytorch lightning trainer.

        :return: a pytorch lightning trainer object
        """
        config = self.trainer_config
        config = _check_pyl_trainer_config(config)
        callbacks: List[Callback] = []
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

    def train(self, train_loader: DataLoader[Any], valid_loader: Optional[DataLoader[Any]] = None) -> None:
        trainer, ckpt_callback = self.init_trainer()

        model = self._init_model()
        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=valid_loader)
        if valid_loader is not None and is_overridden("validation_step", model):
            model.load_state_dict(torch.load(ckpt_callback.best_model_path)["state_dict"])

        self._current_model = model

    def test(self, loader: DataLoader[Any]) -> Dict[str, float]:
        if not self.is_trained():
            raise ValueError("No current model, call 'train(train_loader, valid_loader)' to train the model first")
        trainer, _ = self.init_trainer()
        ret: Dict[str, float] = trainer.test(self._current_model, dataloaders=loader)[0]
        return ret

    def __call__(self, loader: DataLoader[Any]) -> torch.Tensor:
        """
        Call function which outputs model predictions from dataloader

        :param loader: pytorch dataloader
        :return: model predictions of shape (number of samples in loader,1)
        """
        if not self.is_trained():
            raise ValueError("No current model, call 'train(train_loader, valid_loader)' to train the model first")
        model = cast(LightningModule, self._current_model).to(self.device)
        model.eval()

        with torch.no_grad():
            model_prediction = []
            for x, _ in loader:
                x = x.to(self.device)
                model_prediction.append(model(x).detach().cpu())
        predictions = torch.cat(model_prediction, 0)
        return predictions


def _check_pyl_trainer_config(config: Dict[str, Any]) -> Dict[str, Any]:
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
