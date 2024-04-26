from typing import Dict


def config_to_string(config: Dict[str, str]) -> str:
    config_string = ""
    for key, value in config.items():
        config_string += f"{key}+{value}-"
    config_string = config_string.rstrip("+")
    config_string = config_string.rstrip("-")
    return config_string
