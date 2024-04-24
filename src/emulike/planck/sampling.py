import os
from datetime import datetime
import logging
from typing import Tuple, Any
import numpy as np
from ml_collections.config_dict import ConfigDict
import emcee

# our scripts
from src.emulike.planck.distribution import planck_logpost_sampler, planck_priors_normal
from utils.helpers import pickle_save, get_planck_fname
from src.emulike.planck.training import get_training_points, train_gp
from utils.helpers import pickle_load
from src.emulike.planck.accuracy import calculate_planck_accuracy
from experiments.planck.plite import PlanckLitePy

LOGGER = logging.getLogger(__name__)
PATH = os.path.dirname(os.path.realpath(__file__))


def get_priors_emulator(cfg: ConfigDict):
    """
    Generate the priors and get the emulator. See config file for further details. We can
    1) generate the training points
    2) train the GP
    3) load the emulator
    4) Calculate the accuracy

    Args:
        cfg (ConfigDict): the main configuration file

    Returns:
        the priors (uniform or multivariate) and the emulator
    """
    emulator = None

    priors = planck_priors_normal(cfg)
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
        start_time = datetime.now()
        _ = calculate_planck_accuracy(cfg, emulator)
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

    priors, emulator = get_priors_emulator(cfg)
    likelihood = PlanckLitePy(
        data_directory=cfg.path.data,
        year=cfg.planck.year,
        spectra=cfg.planck.spectra,
        use_low_ell_bins=cfg.planck.use_low_ell_bins,
    )

    if cfg.sampling.run_sampler:
        pos = cfg.sampling.mean + 1e-4 * np.random.normal(size=(2 * cfg.ndim, cfg.ndim))
        nwalkers = pos.shape[0]
        start_time = datetime.now()
        sampler = emcee.EnsembleSampler(
            nwalkers,
            cfg.ndim,
            planck_logpost_sampler,
            args=(likelihood, cfg, priors, emulator),
        )
        sampler.run_mcmc(pos, cfg.sampling.nsamples, progress=True)

        time_elapsed = datetime.now() - start_time
        LOGGER.info(f"Time: sample the posterior : {time_elapsed}")

        # save the sampler
        fname = get_planck_fname(cfg)
        path = os.path.join(PATH, "samples")
        pickle_save(sampler, path, fname)
        LOGGER.info(f"Total number of samples: {sampler.flatchain.shape}")
