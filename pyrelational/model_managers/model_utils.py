from typing import Any, Dict

import torch
from lightning.pytorch.trainer.connectors.accelerator_connector import (
    _AcceleratorConnector,
)


def _determine_device(trainer_config: Dict[str, Any]) -> torch.device:
    """
    Determines the torch device of the model from the arguments for the pytorch lightning trainer

    :param trainer_config: configuration dictionary for a pytorch lightning Trainer
    :return: torch device object
    """
    accelerator = _AcceleratorConnector(
        accelerator=trainer_config.get("accelerator", "cpu"), devices=trainer_config.get("devices", "auto")
    )
    device: torch.device = accelerator.strategy.root_device
    return device
