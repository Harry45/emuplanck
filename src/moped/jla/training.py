"""
Code: Main script for training the JLA emulator.
Date: March 2024
Author: Arrykrishna
"""

import os
import logging
from typing import Tuple
import numpy as np
import pandas as pd
from scipy.stats import norm
from ml_collections.config_dict import ConfigDict

# our scripts and functions
from src.emulike.jla.training import input_points_normal
from src.emulike.jla.distribution import jla_priors_normal
from src.moped.jla.functions import JLAmoped, jla_moped_coefficients
from src.moped.jla.emulator import JLAMOPEDemu
from utils.helpers import pickle_save, pickle_load


LOGGER = logging.getLogger(__name__)
PATH = os.path.dirname(os.path.realpath(__file__))


def get_training_points(cfg: ConfigDict) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate the training points (inputs, outputs) for the emulator.

    Args:
        cfg (ConfigDict): the main configuration file.

    Returns:
        Tuple[np.ndarray, np.ndarray]: the cosmological parameters and the log-likelihood values.
    """
    path = os.path.join(PATH, "trainingpoints")
    JLAcompression = JLAmoped(cfg)
    priors = jla_priors_normal(cfg)
    cosmologies = input_points_normal(cfg, priors)
    model = "lcdm" if cfg.lambdacdm else "wcdm"
    fcosmo = f"cosmologies_{model}_{cfg.emu.nlhs}"
    fmoped = f"moped_coefficients_{model}_{cfg.emu.nlhs}"

    LOGGER.info(f"Generating {cfg.emu.nlhs} training points")
    coefficients = jla_moped_coefficients(JLAcompression, cosmologies, cfg)

    pickle_save(cosmologies, path, fcosmo)
    pickle_save(coefficients, path, fmoped)
    return cosmologies, coefficients


def train_gp(cfg: ConfigDict) -> list:
    """
    Train and store the emulator.

    Args:
        cfg (ConfigDict): the main configuration file with all the settings.

    Returns:
        list: a list of emulators
    """
    # path for storing the emulators
    path_emu = os.path.join(PATH, "emulators")

    # path for the training points
    path_tp = os.path.join(PATH, "trainingpoints")

    # file names
    model = "lcdm" if cfg.lambdacdm else "wcdm"
    fcosmo = f"cosmologies_{model}_{cfg.emu.nlhs}"
    fmoped = f"moped_coefficients_{model}_{cfg.emu.nlhs}"

    # training points
    cosmologies = pickle_load(path_tp, fcosmo)
    coefficients = pickle_load(path_tp, fmoped)

    # train and store the emulators
    emulators = {}
    for i in range(cfg.ndim):
        LOGGER.info(f"Training MOPED emulator {i+1}")
        femu = f"emulator_{i}_{model}_{cfg.emu.nlhs}"
        emulator = JLAMOPEDemu(cfg, cosmologies, coefficients[:, i])
        loss = emulator.train_gp()
        pickle_save(emulator, path_emu, femu)
        emulators[i] = emulator
    return emulators
