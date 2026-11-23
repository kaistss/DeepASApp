import os
import sys
import datetime
import logging
import importlib
from typing import Dict

import yaml


# Class order used by the ASA dataset labels (index == class id).
LABELS = {
    'ASA2': [
        'DomesticSound', 'MaleSpeech', 'MusicalInstrument', 'Clapping',
        'Music', 'Laughter', 'WalkAndFootsteps', 'Knock',
        'WaterTap', 'Door', 'FemaleSpeech', 'Bell', 'Telephone'
    ]
}


def ignore_warnings():
    import warnings
    warnings.filterwarnings("ignore", message=".*invalid value encountered in cast.*")
    warnings.filterwarnings("ignore", message=".*deprecated.*")
    warnings.filterwarnings("ignore", message=".*torch.meshgrid.*")
    warnings.filterwarnings("ignore", message=".*that has Tensor Cores.*")
    warnings.filterwarnings("ignore", message=".*cudnnFinalize Descriptor Failed.*")


logging_levels = {
    "CRITICAL": logging.CRITICAL,
    "ERROR": logging.ERROR,
    "WARNING": logging.WARNING,
    "INFO": logging.INFO,
    "DEBUG": logging.DEBUG,
    "NOTSET": logging.NOTSET
}


def logging_setup(directory, file_log_level="INFO", console_log_level="DEBUG"):
    file_log_level = logging_levels[file_log_level]
    console_log_level = logging_levels[console_log_level]
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)

    os.makedirs(directory, exist_ok=True)
    now = datetime.datetime.now()
    log_full_path = os.path.join(directory, now.strftime('%Y%m%d_%Hh%M.log'))

    file_handler = logging.FileHandler(log_full_path)
    file_handler.setLevel(file_log_level)
    file_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s'))
    root_logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_log_level)
    console_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(name)s (%(processName)s): %(message)s'))
    root_logger.addHandler(console_handler)


def parse_yaml(config_yaml: str) -> Dict:
    with open(config_yaml, "r") as fr:
        return yaml.load(fr, Loader=yaml.FullLoader)


def initialize_config(module_cfg, reload=False):
    """Instantiate `module_cfg['main']` from `module_cfg['module']` with `module_cfg['args']`."""
    if reload and module_cfg["module"] in sys.modules:
        module = importlib.reload(sys.modules[module_cfg["module"]])
    else:
        module = importlib.import_module(module_cfg["module"])
    if 'args' in module_cfg.keys():
        return getattr(module, module_cfg["main"])(**module_cfg["args"])
    return getattr(module, module_cfg["main"])()
