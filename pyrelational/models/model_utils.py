import warnings
from typing import List, Union

import torch


def _determine_device(gpus: Union[List[int], str, int, None]) -> torch.device:
    """
    Determines the torch device of the model from the gpus argument for pytorch lightning trainer

    :param gpus: Number of gpus (int) or which gpus to train on (str, list)
    :return: torch device
    """
    if isinstance(gpus, list):
        gpus = str(gpus[0])
        warnings.warn("Multiple GPUs provided, setting the first GPU to be device used in call function of model")
        return torch.device(f"cuda:{gpus}")
    elif isinstance(gpus, str):
        return torch.device(f"cuda:{gpus}")
    elif isinstance(gpus, int) and (gpus > 0):
        return torch.device("cuda")
    else:
        return torch.device("cpu")
