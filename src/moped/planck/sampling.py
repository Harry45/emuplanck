import os
import logging
from datetime import datetime
from typing import Tuple, Any
import numpy as np
from ml_collections.config_dict import ConfigDict
import emcee
from multiprocessing import Pool

# our scripts and functions
from torchemu.gaussianprocess import GaussianProcess
from src.moped.planck.training import get_training_points, train_gp
from src.moped.planck.accuracy import planck_moped_accuracy
from src.emulike.planck.distribution import planck_priors_normal, planck_logprior_normal
from src.moped.planck.functions import PLANCKmoped, planck_moped_coefficients
from src.moped.planck.emulator import PlanckMOPEDemu
from experiments.planck.model import planck_get_params
from utils.helpers import pickle_save, pickle_load, get_planck_fname


LOGGER = logging.getLogger(__name__)
PATH = os.path.dirname(os.path.realpath(__file__))


def get_planck_priors_emulator(cfg: ConfigDict) -> Tuple[dict, GaussianProcess]:
    """
    Generate the priors and get the emulator. See config file for further details. We can
    1) generate the training points
    2) train the GP
    3) load the emulator
    4) Calculate the accuracy

    Args:
        cfg (ConfigDict): the main configuration file

    Returns:
        Tuple[dict, GaussianProcess]: the priors and the emulator
    """
    emulators = None
    priors = planck_priors_normal(cfg)
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
        _ = planck_moped_accuracy(cfg, emulators)
        time_elapsed = datetime.now() - start_time
        LOGGER.info(f"Time: Accuracy for {cfg.emu.ntest} points : {time_elapsed}")

    return priors, emulators


def planck_loglike_moped_sampler(
    parameters: np.ndarray,
    compressor: PLANCKmoped,
    cfg: ConfigDict,
    priors: Any,
    emulators: list = None,
) -> np.ndarray:
    """
    Calculates the log-likelihood using the emulator or the simulator.

    Args:
        parameters (np.ndarray): the input parameters
        compressor (PLANCKmoped): the compressor
        cfg (ConfigDict): the main configuration file
        priors (Any): the priors on the cosmological parameters
        emulators (list): the pre-trained emulators

    Returns:
        np.ndarray: the log-likelihood value
    """
    logprior = planck_logprior_normal(parameters, priors)

    if np.isfinite(logprior):
        if cfg.sampling.use_gp:
            coef = [emulators[i].prediction(parameters) for i in range(cfg.ndim)]
            coef = np.asarray(coef)
        else:
            coef = planck_moped_coefficients(compressor, parameters, cfg)

        diff = compressor.store.ycomp - coef.reshape(-1)
        return -0.5 * sum(diff**2)
    return -1e32


def planck_logpost_moped_sampler(
    parameters: np.ndarray,
    compressor: PLANCKmoped,
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
    loglike = planck_loglike_moped_sampler(
        parameters, compressor, cfg, priors, emulators
    )
    logprior = planck_logprior_normal(parameters, priors)
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

    priors, emulator = get_planck_priors_emulator(cfg)
    compressor = PLANCKmoped(cfg)

    if cfg.sampling.run_sampler:
        pos = cfg.sampling.mean + 1e-4 * np.random.normal(size=(2 * cfg.ndim, cfg.ndim))
        nwalkers = pos.shape[0]
        start_time = datetime.now()

        # this does not seem to work with MOPED
        # with Pool() as pool:
        #     sampler = emcee.EnsembleSampler(
        #         nwalkers,
        #         cfg.ndim,
        #         planck_logpost_moped_sampler,
        #         args=(compressor, cfg, priors, emulator),
        #         pool=pool,
        #     )
        #     sampler.run_mcmc(pos, cfg.sampling.nsamples, progress=True)

        sampler = emcee.EnsembleSampler(
            nwalkers,
            cfg.ndim,
            planck_logpost_moped_sampler,
            args=(compressor, cfg, priors, emulator),
        )
        sampler.run_mcmc(pos, cfg.sampling.nsamples, progress=True)

        time_elapsed = datetime.now() - start_time
        LOGGER.info(f"Time: sample the posterior : {time_elapsed}")

        # save the sampler
        fname = get_planck_fname(cfg)
        path = os.path.join(PATH, "samples")
        pickle_save(sampler, path, fname)
        pickle_save(sampler, path, fname)
        LOGGER.info(f"Total number of samples: {sampler.flatchain.shape}")
