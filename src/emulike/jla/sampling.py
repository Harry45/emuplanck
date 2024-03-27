import os
import logging
from datetime import datetime
from typing import Tuple, Any
import numpy as np
from ml_collections.config_dict import ConfigDict
import emcee
from multiprocessing import Pool

# our scripts and functions
from experiments.jla.jlalite import JLALitePy
from torchemu.gaussianprocess import GaussianProcess
from utils.helpers import pickle_load, pickle_save, get_jla_fname
from src.emulike.jla.training import get_training_points, train_gp
from src.emulike.jla.accuracy import calculate_jla_accuracy
from src.emulike.jla.distribution import jla_priors_normal, jla_logpost_sampler


LOGGER = logging.getLogger(__name__)
PATH = os.path.dirname(os.path.realpath(__file__))


def get_jla_priors_emulator(cfg: ConfigDict) -> Tuple[dict, GaussianProcess]:
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
    emulator = None
    priors = jla_priors_normal(cfg)
    path_emu = os.path.join(PATH, "emulators")
    model = "lcdm" if cfg.lambdacdm else "wcdm"
    femu = f"emulator_{model}_{cfg.emu.nlhs}"

    if cfg.emu.generate_points:
        start_time = datetime.now()
        _ = get_training_points(cfg)
        time_elapsed = datetime.now() - start_time
        LOGGER.info(f"Time: generate {cfg.emu.nlhs} training points : {time_elapsed}")

    if cfg.emu.train_emu:
        start_time = datetime.now()
        emulator = train_gp(cfg)
        time_elapsed = datetime.now() - start_time
        LOGGER.info(f"Time: training : {cfg.emu.nlhs} training points : {time_elapsed}")

    if cfg.sampling.use_gp:
        emulator = pickle_load(path_emu, femu)

    if cfg.emu.calc_acc:
        emulator = pickle_load(path_emu, femu)
        start_time = datetime.now()
        _ = calculate_jla_accuracy(cfg, emulator)
        time_elapsed = datetime.now() - start_time
        LOGGER.info(f"Time: Accuracy for {cfg.emu.ntest} points : {time_elapsed}")

    return priors, emulator


def sample_posterior(cfg: ConfigDict) -> emcee.ensemble.EnsembleSampler:
    """
    Sample the posterior distribution.

    Args:
        cfg (ConfigDict): the main configuration file

    Returns:
        emcee.ensemble.EnsembleSampler: the EMCEE sampler
    """

    priors, emulator = get_jla_priors_emulator(cfg)
    likelihood = JLALitePy(cfg)

    if cfg.sampling.run_sampler:
        pos = cfg.sampling.mean + 1e-4 * np.random.normal(size=(2 * cfg.ndim, cfg.ndim))
        nwalkers = pos.shape[0]
        start_time = datetime.now()
        # with Pool() as pool:
        #     sampler = emcee.EnsembleSampler(
        #         nwalkers,
        #         cfg.ndim,
        #         jla_logpost_sampler,
        #         args=(likelihood, cfg, priors, emulator),
        #         pool=pool,
        #     )
        #     sampler.run_mcmc(pos, cfg.sampling.nsamples, progress=True)
        sampler = emcee.EnsembleSampler(
            nwalkers,
            cfg.ndim,
            jla_logpost_sampler,
            args=(likelihood, cfg, priors, emulator),
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
