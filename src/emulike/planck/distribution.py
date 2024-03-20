"""
Code: Sampler for the Planck Lite likelihood code.
Date: August 2023
Author: Arrykrishna
"""

# pylint: disable=bad-continuation
from typing import Any
import numpy as np
import logging
import scipy.stats as ss
from ml_collections.config_dict import ConfigDict
from scipy.stats import multivariate_normal


# our scripts
from experiments.planck.model import planck_loglike
from torchemu.gaussianprocess import GaussianProcess
from experiments.planck.plite import PlanckLitePy

LOGGER = logging.getLogger(__name__)


def planck_priors_normal(cfg: ConfigDict) -> dict:
    """
    Generate normal priors on the cosmological parameters.

    Args:
        cfg (ConfigDict): the main configuration file.

    Returns:
        dict: a dictionary of the priors of the parameters.
    """
    priors = {}
    for i, name in enumerate(cfg.sampling.names):
        loc = cfg.sampling.mean[i]
        scale = cfg.sampling.std[i]
        priors[name] = ss.norm(loc, scale)
    return priors


def planck_logprior_normal(parameters: np.ndarray, priors: dict) -> float:
    """
    Calculates the log-prior, where all parameters follow a uniform distribution.

    Args:
        parameters (np.ndarray): the input parameter.
        priors (dict): a dictionary of priors for the parameters.

    Returns:
        float: the log-prior.
    """
    logp = 0.0
    for i, name in enumerate(priors):
        logp += priors[name].logpdf(parameters[i])
    return logp


def planck_loglike_sampler(
    parameters: np.ndarray,
    likelihood: PlanckLitePy,
    cfg: ConfigDict,
    priors: Any,
    emulator: GaussianProcess = None,
) -> np.ndarray:
    """
    Calculates the log-likelihood using the emulator or the simulator.

    Args:
        parameters (np.ndarray): the input parameters
        cfg (ConfigDict): the main configuration file
        priors (Any): the priors on the cosmological parameters
        emulator (GaussianProcess): the pre-trained emulator

    Returns:
        np.ndarray: the log-likelihood value
    """
    logprior = planck_logprior_normal(parameters, priors)

    if np.isfinite(logprior):
        if cfg.sampling.use_gp:
            loglike = emulator.prediction(parameters)
        else:
            loglike = planck_loglike(likelihood, parameters, cfg)
        return loglike
    return -1e32


def planck_logpost_sampler(
    parameters: np.ndarray,
    likelihood: PlanckLitePy,
    cfg: ConfigDict,
    priors: Any,
    emulator: GaussianProcess = None,
) -> float:
    """
    the log-posterior calculated either with the emulator or the simulator.

    Args:
        parameters (np.ndarray): the vector of parameters.
        cfg (ConfigDict): the main configuration file.
        priors (Any): the priors on the cosmological parameters.
        emulator (GaussianProcess): the pre-trained emulator (defaults to None)

    Returns:
        float: the log-posterior value
    """

    loglike = planck_loglike_sampler(parameters, likelihood, cfg, priors, emulator)
    logprior = planck_logprior_normal(parameters, priors)

    logpost = loglike + logprior
    if np.isfinite(logpost):
        return logpost.item()
    return -1e32
