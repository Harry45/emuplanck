"""
Code: Main script for training the emulators.
Date: August 2023
Author: Arrykrishna
"""

import os
from typing import Tuple
import numpy as np
from ml_collections.config_dict import ConfigDict

# our scripts and functions
from src.cambrun import calculate_loglike
from src.helpers import pickle_save, pickle_load
from src.sampling import generate_priors_uniform, generate_priors_multivariate
from src.emulator import PlanckEmu, input_points_uniform, input_points_multivariate


def get_training_points(cfg: ConfigDict) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate the training points (inputs, outputs) for the emulator.

    Args:
        cfg (ConfigDict): the main configuration file.

    Returns:
        Tuple[np.ndarray, np.ndarray]: the cosmological parameters and the log-likelihood values.
    """
    os.makedirs("training", exist_ok=True)

    if cfg.sampling.uniform_prior:
        priors = generate_priors_uniform(cfg)
        cosmologies = input_points_uniform(cfg, priors, cfg.emu.nlhs)
        fcosmo = f"cosmologies_uniform_{cfg.emu.nlhs}"
        flike = f"loglike_uniform_{cfg.emu.nlhs}"
    else:
        priors = generate_priors_multivariate(cfg)
        cosmologies = input_points_multivariate(cfg, priors, cfg.emu.nlhs)
        fcosmo = f"cosmologies_multivariate_{cfg.emu.nlhs}"
        flike = f"loglike_multivariate_{cfg.emu.nlhs}"

    print(f"Generating {cfg.emu.nlhs} training points")
    loglikelihoods = calculate_loglike(cosmologies, cfg)

    pickle_save(cosmologies, "training", fcosmo)
    pickle_save(loglikelihoods, "training", flike)
    return cosmologies, loglikelihoods


def train_gp(cfg: ConfigDict) -> PlanckEmu:
    """
    Train and store the emulator.

    Args:
        cfg (ConfigDict): the main configuration file with all the settings.

    Returns:
        PlanckEmu: the emulator
    """
    os.makedirs("emulators", exist_ok=True)

    if cfg.sampling.uniform_prior:
        fcosmo = f"cosmologies_uniform_{cfg.emu.nlhs}"
        flike = f"loglike_uniform_{cfg.emu.nlhs}"
        femu = f"emulator_uniform_{cfg.emu.nlhs}"
    else:
        fcosmo = f"cosmologies_multivariate_{cfg.emu.nlhs}"
        flike = f"loglike_multivariate_{cfg.emu.nlhs}"
        femu = f"emulator_multivariate_{cfg.emu.nlhs}"

    cosmologies = pickle_load("training", fcosmo)
    loglikelihoods = pickle_load("training", flike)

    emulator = PlanckEmu(cfg, cosmologies, loglikelihoods)
    _ = emulator.train_gp(prewhiten=True)
    pickle_save(emulator, "emulators", femu)
    return emulator
