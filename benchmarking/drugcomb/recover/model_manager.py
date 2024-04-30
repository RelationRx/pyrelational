from typing import Any, Dict, List, Union, cast

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from benchmarking.drugcomb.recover.module import RecoverModel
from pyrelational.model_managers import LightningModelManager


class RecoverModelManager(LightningModelManager):

    def __init__(
        self,
        model_config: Union[Dict[str, Any], str],
        trainer_config: Union[Dict[str, Any], str],
    ):
        super().__init__(RecoverModel, model_config=model_config, trainer_config=trainer_config)

    def __call__(self, loader: DataLoader[Any]) -> Tensor:
        if not self.is_trained():
            raise ValueError("No current model, call 'train(train_loader, valid_loader)' to train the model first")
        trainer, _ = self.init_trainer()
        preds = cast(List[Tensor], trainer.predict(self._current_model, dataloaders=loader))
        return torch.cat(preds)
