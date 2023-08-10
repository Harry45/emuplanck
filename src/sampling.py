"""
Code: Sampler for the Planck Lite likelihood code.
Date: August 2023
Author: Arrykrishna
"""
# pylint: disable=bad-continuation
from typing import Any
import numpy as np
import emcee
import scipy.stats as ss
from ml_collections.config_dict import ConfigDict
from scipy.stats import multivariate_normal

# our scripts
from src.cambrun import calculate_loglike
from src.torchemu.gaussianprocess import GaussianProcess


def generate_priors_uniform(cfg: ConfigDict) -> dict:
    """
    Generate uniform priors on the cosmological parameters.

    Args:
        cfg (ConfigDict): the main configuration file.

    Returns:
        dict: a dictionary of the priors of the parameters.
    """
    priors = {}
    for i, name in enumerate(cfg.cosmo.names):
        priors[name] = ss.uniform(
            cfg.sampling.mean[i] - cfg.sampling.nstd * cfg.sampling.std[i],
            2.0 * cfg.sampling.nstd * cfg.sampling.std[i],
        )
    return priors


def generate_priors_multivariate(cfg: ConfigDict) -> multivariate_normal:
    """
    Generates a multivariate normal distribution  of the parameters with a mean and covariance, C

    Args:
        cfg (ConfigDict): the main configuration file

    Returns:
        multivariate_normal: the multivariate normal distribution on the parameters.
    """
    return multivariate_normal(cfg.sampling.mean, cfg.sampling.ncov * cfg.sampling.cov)


def emcee_logprior_multivariate(
    parameters: np.ndarray, priors: multivariate_normal
) -> float:
    """
    Calculates the log-pdf using a multivariate normal prior.

    Args:
        parameters (np.ndarray): a parameter vector
        priors (multivariate_normal): the multivariate normal prior

    Returns:
        float: the log-pdf
    """
    logp = priors.logpdf(parameters)
    return logp


def emcee_logprior_uniform(parameters: np.ndarray, priors: dict) -> float:
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


def emcee_loglike(
    parameters: np.ndarray, cfg: ConfigDict, emulator: GaussianProcess = None
) -> np.ndarray:
    """
    Calculates the log-likelihood using the emulator or the simulator.

    Args:
        parameters (np.ndarray): the input parameters
        cfg (ConfigDict): the main configuration file
        emulator (GaussianProcess): the pre-trained emulator

    Returns:
        np.ndarray: the log-likelihood value
    """
    if cfg.sampling.use_gp:
        loglike = emulator.prediction(parameters)
    else:
        loglike = calculate_loglike(parameters, cfg)
    return loglike


def emcee_logpost(
    parameters: np.ndarray,
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

    loglike = emcee_loglike(parameters, cfg, emulator)

    if cfg.sampling.uniform_prior:
        logprior = emcee_logprior_uniform(parameters, priors)
    else:
        logprior = emcee_logprior_multivariate(parameters, priors)

    logpost = loglike + logprior
    if np.isfinite(logpost):
        return logpost.item()
    return -1e32
