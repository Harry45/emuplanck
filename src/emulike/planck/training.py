"""
Code: Main script for training the emulators.
Date: August 2023
Author: Arrykrishna
"""

import os
import logging
from typing import Any
from typing import Tuple
import numpy as np
import pandas as pd
from scipy.stats import norm
from ml_collections.config_dict import ConfigDict

# our scripts and functions
from experiments.planck.plite import PlanckLitePy
from experiments.planck.model import planck_loglike
from utils.helpers import pickle_save, pickle_load
from src.emulike.planck.emulator import PlanckEmu
from src.emulike.planck.distribution import (
    planck_priors_uniform,
    planck_priors_multivariate,
)


LOGGER = logging.getLogger(__name__)
PATH = os.path.dirname(os.path.realpath(__file__))


def input_points_multivariate(cfg: ConfigDict, priors: Any) -> np.ndarray:
    """
    Transform the input LH points such that they follow a multivariate normal distribution.

    Args:
        cfg (ConfigDict): the main configuration file
        priors (Any): the multivariate normal prior.

    Returns:
        np.ndarray: the transformed LH points.
    """
    fname = os.path.join(cfg.path.parent, f"lhs/samples_{cfg.ndim}_{cfg.emu.nlhs}.csv")
    lhs_samples = pd.read_csv(fname, index_col=0)
    dist = norm(0, 1)
    scaled_samples = []
    for i in range(cfg.ndim):
        scaled_samples.append(dist.ppf(lhs_samples.values[:, i]))
    scaled_samples = np.column_stack(scaled_samples)
    cholesky = np.linalg.cholesky(priors.cov)
    scaled = cfg.sampling.mean.reshape(cfg.ndim, 1) + cholesky @ scaled_samples.T
    scaled = scaled.T
    return scaled


def input_points_uniform(cfg: ConfigDict, priors: dict) -> np.ndarray:
    """
    Transform the input LH according to hypercube (uniform in each direction)

    Args:
        cfg (ConfigDict): the main configuration file
        priors (dict): a list of uniform priors for the cosmological parameters

    Returns:
        np.ndarray: the scaled LH points.
    """
    fname = os.path.join(cfg.path.parent, f"lhs/samples_{cfg.ndim}_{cfg.emu.nlhs}.csv")
    lhs_samples = pd.read_csv(fname, index_col=0)
    scaled_samples = []
    for i, name in enumerate(cfg.cosmo.names):
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
    likelihood = PlanckLitePy(
        data_directory=cfg.path.data,
        year=cfg.planck.year,
        spectra=cfg.planck.spectra,
        use_low_ell_bins=cfg.planck.use_low_ell_bins,
    )

    if cfg.sampling.uniform_prior:
        priors = planck_priors_uniform(cfg)
        cosmologies = input_points_uniform(cfg, priors)
        fcosmo = f"cosmologies_uniform_{cfg.emu.nlhs}"
        flike = f"loglike_uniform_{cfg.emu.nlhs}"
    else:
        priors = planck_priors_multivariate(cfg)
        cosmologies = input_points_multivariate(cfg, priors)
        fcosmo = f"cosmologies_multivariate_{cfg.emu.nlhs}"
        flike = f"loglike_multivariate_{cfg.emu.nlhs}"

    LOGGER.info(f"Generating {cfg.emu.nlhs} training points")
    loglikelihoods = planck_loglike(likelihood, cosmologies, cfg)

    pickle_save(cosmologies, path, fcosmo)
    pickle_save(loglikelihoods, path, flike)
    return cosmologies, loglikelihoods


def train_gp(cfg: ConfigDict) -> PlanckEmu:
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

    if cfg.sampling.uniform_prior:
        fcosmo = f"cosmologies_uniform_{cfg.emu.nlhs}"
        flike = f"loglike_uniform_{cfg.emu.nlhs}"
        femu = f"emulator_uniform_{cfg.emu.nlhs}"
    else:
        fcosmo = f"cosmologies_multivariate_{cfg.emu.nlhs}"
        flike = f"loglike_multivariate_{cfg.emu.nlhs}"
        femu = f"emulator_multivariate_{cfg.emu.nlhs}"

    cosmologies = pickle_load(path_tp, fcosmo)
    loglikelihoods = pickle_load(path_tp, flike)

    emulator = PlanckEmu(cfg, cosmologies, loglikelihoods)
    _ = emulator.train_gp(prewhiten=True)
    pickle_save(emulator, path_emu, femu)
    return emulator
