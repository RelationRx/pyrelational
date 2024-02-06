import warnings
from typing import Any, Dict, List, Union

import torch


def _determine_device(trainer_config: Dict[str, Any]) -> torch.device:
    """
    Determines the torch device of the model from the gpus argument for pytorch lightning trainer

    :param trainer_config: configuration dictionary for a pytorch lightning Trainer
    :return: torch device
    """
    if trainer_config.get("accelerator", "cpu") == "cpu":
        return torch.device("cpu")
    else:
        devices: Union[List[int], str, int] = trainer_config.get("devices", "auto")
        if isinstance(devices, str):
            return torch.device("cuda:0")
        elif isinstance(devices, int):
            i = 0 if devices == -1 else devices  # device==-1 means using all available devices
            return torch.device(f"cuda:{i}")
        else:
            warnings.warn("Multiple GPUs provided, pyrelational will use the first one in the model's __call__ method.")
            return torch.device(f"cuda:{devices[0]}")
