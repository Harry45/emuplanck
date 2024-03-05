"""
Author: Arrykrishna
Date: January 2023
Email: arrykrish@gmail.com
Project: Inference of bias parameters.
"""

import os
import logging
import sys
from datetime import datetime
from ml_collections.config_dict import ConfigDict


NOW = datetime.now()
FORMATTER = logging.Formatter("[%(levelname)s] - %(asctime)s - %(name)s : %(message)s")
CONSOLE_FORMATTER = logging.Formatter("[%(levelname)s]: %(message)s")
DATETIME = NOW.strftime("%d-%m-%Y-%H-%M")


def get_logger(config: ConfigDict) -> logging.Logger:
    """Generates a logging file for storing all information.
    Args:
        config (ConfigDict): The main configuration file
    Returns:
        logging.Logger: the logging module
    """
    # create the folder if it does not exist
    os.makedirs(config.path.logs, exist_ok=True)

    fname = config.path.logs + config.logname + f"_{DATETIME}.log"
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.propagate = False

    fhandler = logging.FileHandler(filename=fname)
    fhandler.setLevel(logging.DEBUG)
    fhandler.setFormatter(FORMATTER)

    logger.addHandler(fhandler)

    return logger


def get_logger_terminal(config: ConfigDict) -> logging.Logger:
    """Generates a logging file for storing all information.
    Args:
        config (ConfigDict): The main configuration file
    Returns:
        logging.Logger: the logging module
    """
    fname = config.path.logs + config.logname + f"_{DATETIME}.log"
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.propagate = False

    chandler = logging.StreamHandler(sys.stdout)
    chandler.setLevel(logging.DEBUG)
    chandler.setFormatter(CONSOLE_FORMATTER)

    fhandler = logging.FileHandler(filename=fname)
    fhandler.setLevel(logging.DEBUG)
    fhandler.setFormatter(FORMATTER)

    logger.addHandler(fhandler)
    logger.addHandler(chandler)
    return logger
