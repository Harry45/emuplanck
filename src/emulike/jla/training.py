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
from src.emulike.jla.emulator import JLAemu
from src.emulike.jla.distribution import jla_priors_normal
from experiments.jla.jlalite import JLALitePy
from experiments.jla.model import jla_loglike
from utils.helpers import pickle_save, pickle_load

LOGGER = logging.getLogger(__name__)
PATH = os.path.dirname(os.path.realpath(__file__))


def input_points_normal(cfg: ConfigDict, priors: dict) -> np.ndarray:
    """
    Transform the input LH according to hypercube (uniform in each direction)

    Args:
        cfg (ConfigDict): the main configuration file
        priors (dict): a list of uniform priors for the cosmological parameters

    Returns:
        np.ndarray: the scaled LH points.
    """
    LOGGER.info("Generating LHS Points")
    os.system(f"Rscript sampleLHS.R {cfg.emu.nlhs} {cfg.ndim}")
    fname = os.path.join(cfg.path.parent, f"lhs/samples_{cfg.ndim}_{cfg.emu.nlhs}.csv")
    lhs_samples = pd.read_csv(fname, index_col=0)
    scaled_samples = []
    for i, name in enumerate(cfg.sampling.names):
        scaled_samples.append(priors[name].ppf(lhs_samples.values[:, i]))
    scaled_samples = np.column_stack(scaled_samples)
    return scaled_samples


def get_training_points(cfg: ConfigDict) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate the training points (inputs, outputs) for the emulator.

    Args:
        cfg (ConfigDict): the main configuration file.

    Returns:
        Tuple[np.ndarray, np.ndarray]: the cosmological parameters and the log-likelihood values.
    """
    path = os.path.join(PATH, "trainingpoints")
    likelihood = JLALitePy(cfg)
    priors = jla_priors_normal(cfg)
    cosmologies = input_points_normal(cfg, priors)
    model = "lcdm" if cfg.lambdacdm else "wcdm"
    fcosmo = f"cosmologies_{model}_{cfg.emu.nlhs}"
    flike = f"loglike_{model}_{cfg.emu.nlhs}"

    LOGGER.info(f"Generating {cfg.emu.nlhs} training points")
    loglikelihoods = jla_loglike(likelihood, cosmologies, cfg)

    pickle_save(cosmologies, path, fcosmo)
    pickle_save(loglikelihoods, path, flike)
    return cosmologies, loglikelihoods


def train_gp(cfg: ConfigDict) -> JLAemu:
    """
    Train and store the emulator.

    Args:
        cfg (ConfigDict): the main configuration file with all the settings.

    Returns:
        PlanckEmu: the emulator
    """
    # path for storing the emulators
    path_emu = os.path.join(PATH, "emulators")

    # path for the training points
    path_tp = os.path.join(PATH, "trainingpoints")

    # file names
    model = "lcdm" if cfg.lambdacdm else "wcdm"
    fcosmo = f"cosmologies_{model}_{cfg.emu.nlhs}"
    flike = f"loglike_{model}_{cfg.emu.nlhs}"
    femu = f"emulator_{model}_{cfg.emu.nlhs}"

    # training points
    cosmologies = pickle_load(path_tp, fcosmo)
    loglikelihoods = pickle_load(path_tp, flike)

    # train and store the emulators
    emulator = JLAemu(cfg, cosmologies, loglikelihoods)
    loss = emulator.train_gp()
    pickle_save(emulator, path_emu, femu)
    return emulator
