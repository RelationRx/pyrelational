import os
from typing import Dict

import torch
from lightning import seed_everything


def config_to_string(config: Dict[str, str]) -> str:
    config_string = ""
    for key, value in config.items():
        config_string += f"{key}+{value}-"
    config_string = config_string.rstrip("+")
    config_string = config_string.rstrip("-")
    return config_string


def make_reproducible(seed: int) -> None:
    """Ensure everything is reproducible.

    Inspired from lightning Trainer 'deterministic' argument and seed_everything.
    """
    seed_everything(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
