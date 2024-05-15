import os
import logging
from datetime import datetime
from typing import Tuple, Any
import numpy as np
from ml_collections.config_dict import ConfigDict
import emcee

# our scripts and functions
from src.emulike.jla.distribution import jla_priors_normal, jla_logprior_normal
from src.moped.jla.training import get_training_points, train_gp
from src.moped.jla.accuracy import jla_moped_accuracy
from src.moped.jla.functions import JLAmoped, jla_moped_coefficients
from experiments.jla.model import get_jla_params
from utils.helpers import pickle_load, pickle_save, get_jla_fname


LOGGER = logging.getLogger(__name__)
PATH = os.path.dirname(os.path.realpath(__file__))


def get_jla_priors_emulator(cfg: ConfigDict):
    """
    Generate the priors and get the emulator. See config file for further details. We can
    1) generate the training points
    2) train the GP
    3) load the emulator
    4) Calculate the accuracy

    Args:
        cfg (ConfigDict): the main configuration file

    Returns:
        the priors and the emulator
    """
    emulators = None
    priors = jla_priors_normal(cfg)
    path_emu = os.path.join(PATH, "emulators")
    model = "lcdm" if cfg.lambdacdm else "wcdm"

    if cfg.emu.generate_points:
        start_time = datetime.now()
        _ = get_training_points(cfg)
        time_elapsed = datetime.now() - start_time
        LOGGER.info(f"Time: generate {cfg.emu.nlhs} training points : {time_elapsed}")

    if cfg.emu.train_emu:
        start_time = datetime.now()
        emulators = train_gp(cfg)
        time_elapsed = datetime.now() - start_time
        LOGGER.info(f"Time: training : {cfg.emu.nlhs} training points : {time_elapsed}")

    if cfg.sampling.use_gp:
        emulators = {}
        for i in range(cfg.ndim):
            femu = f"emulator_{i}_{model}_{cfg.emu.nlhs}"
            emulators[i] = pickle_load(path_emu, femu)

    if cfg.emu.calc_acc:
        emulators = {}
        for i in range(cfg.ndim):
            femu = f"emulator_{i}_{model}_{cfg.emu.nlhs}"
            emulators[i] = pickle_load(path_emu, femu)
        start_time = datetime.now()
        _ = jla_moped_accuracy(cfg, emulators)
        time_elapsed = datetime.now() - start_time
        LOGGER.info(f"Time: Accuracy for {cfg.emu.ntest} points : {time_elapsed}")

    return priors, emulators


def jla_loglike_moped_sampler(
    parameters: np.ndarray,
    compressor: JLAmoped,
    cfg: ConfigDict,
    priors: Any,
    emulators: list = None,
) -> np.ndarray:
    """
    Calculates the log-likelihood using the emulator or the simulator.

    Args:
        parameters (np.ndarray): the input parameters
        compressor (JLAmoped): the compressor
        cfg (ConfigDict): the main configuration file
        priors (Any): the priors on the cosmological parameters
        emulators (list): the pre-trained emulators

    Returns:
        np.ndarray: the log-likelihood value
    """
    logprior = jla_logprior_normal(parameters, priors)

    if np.isfinite(logprior):
        if cfg.sampling.use_gp:
            coef = [emulators[i].prediction(parameters) for i in range(cfg.ndim)]
            coef = np.asarray(coef)
        else:
            coef = jla_moped_coefficients(compressor, parameters, cfg)
        diff = compressor.store.ycomp - coef.reshape(-1)
        return -0.5 * sum(diff**2)
    return -1e32


def jla_logpost_moped_sampler(
    parameters: np.ndarray,
    compressor: JLAmoped,
    cfg: ConfigDict,
    priors: Any,
    emulators: list = None,
) -> float:
    """
    The log-posterior calculated either with the emulator or the simulator.

    Args:
        parameters (np.ndarray): the input parameters
        compressor (JLAmoped): the compressor
        cfg (ConfigDict): the main configuration file
        priors (Any): the priors on the cosmological parameters
        emulators (list): the pre-trained emulators

    Returns:
        float: the log-posterior value
    """
    loglike = jla_loglike_moped_sampler(parameters, compressor, cfg, priors, emulators)
    logprior = jla_logprior_normal(parameters, priors)
    logpost = loglike + logprior
    if np.isfinite(logpost):
        return logpost.item()
    return -1e32


def sample_posterior(cfg: ConfigDict) -> emcee.ensemble.EnsembleSampler:
    """
    Sample the posterior distribution.

    Args:
        cfg (ConfigDict): the main configuration file

    Returns:
        emcee.ensemble.EnsembleSampler: the EMCEE sampler
    """

    priors, emulator = get_jla_priors_emulator(cfg)
    compressor = JLAmoped(cfg)

    if cfg.sampling.run_sampler:
        pos = cfg.sampling.mean + 1e-4 * np.random.normal(size=(2 * cfg.ndim, cfg.ndim))
        nwalkers = pos.shape[0]
        start_time = datetime.now()
        sampler = emcee.EnsembleSampler(
            nwalkers,
            cfg.ndim,
            jla_logpost_moped_sampler,
            args=(compressor, cfg, priors, emulator),
        )
        sampler.run_mcmc(pos, cfg.sampling.nsamples, progress=True)
        time_elapsed = datetime.now() - start_time
        LOGGER.info(f"Time: sample the posterior : {time_elapsed}")

        # save the sampler
        fname = get_jla_fname(cfg)
        path = os.path.join(PATH, "samples")
        pickle_save(sampler, path, fname)
        pickle_save(sampler, path, fname)
        LOGGER.info(f"Total number of samples: {sampler.flatchain.shape}")
