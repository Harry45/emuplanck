"""
Code: Main script for sampling the Planck Lite likelihood.
Date: August 2023
Author: Arrykrishna
"""
# pylint: disable=bad-continuation
from typing import Tuple, Any
from datetime import datetime
from absl import flags, app
from ml_collections.config_flags import config_flags
from ml_collections.config_dict import ConfigDict
import numpy as np
import emcee

# our scripts and functions
from src.helpers import pickle_save, pickle_load, get_fname
from src.emulator import calculate_accuracy
from src.torchemu.gaussianprocess import GaussianProcess
from src.training import get_training_points, train_gp
from src.sampling import (
    generate_priors_uniform,
    generate_priors_multivariate,
    emcee_logpost,
)


FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", None, "Main configuration file.")


def get_priors_emulator(cfg: ConfigDict) -> Tuple[Any, GaussianProcess]:
    """
    Generate the priors and get the emulator. See config file for further details. We can
    1) generate the training points
    2) train the GP
    3) load the emulator
    4) Calculate the accuracy

    Args:
        cfg (ConfigDict): the main configuration file

    Returns:
        Tuple[Any, GaussianProcess]: the priors (uniform or multivariate) and the emulator
    """
    emulator = None

    if cfg.sampling.uniform_prior:
        priors = generate_priors_uniform(cfg)
        femu = f"emulator_uniform_{cfg.emu.nlhs}"
    else:
        priors = generate_priors_multivariate(cfg)
        femu = f"emulator_multivariate_{cfg.emu.nlhs}"

    if cfg.emu.generate_points:
        start_time = datetime.now()
        _ = get_training_points(cfg)
        time_elapsed = datetime.now() - start_time
        print(
            f"Time taken (hh:mm:ss.ms) to generate {cfg.emu.nlhs} training points : {time_elapsed}"
        )

    if cfg.emu.train_emu:
        start_time = datetime.now()
        emulator = train_gp(cfg)
        time_elapsed = datetime.now() - start_time
        print(
            f"Time taken (hh:mm:ss.ms) to train emulator with {cfg.emu.nlhs} training points : {time_elapsed}"
        )

    if cfg.sampling.use_gp:
        emulator = pickle_load("emulators", femu)

    if cfg.emu.calc_acc:
        start_time = datetime.now()
        _ = calculate_accuracy(cfg, emulator)
        time_elapsed = datetime.now() - start_time
        print(
            f"Time taken (hh:mm:ss.ms) to calculate the accuracy for {cfg.emu.ntest} points : {time_elapsed}"
        )

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

    if cfg.sampling.run_sampler:
        pos = cfg.sampling.mean + 1e-4 * np.random.normal(size=(2 * cfg.ndim, cfg.ndim))
        nwalkers = pos.shape[0]

        sampler = emcee.EnsembleSampler(
            nwalkers, cfg.ndim, emcee_logpost, args=(cfg, priors, emulator)
        )
        start_time = datetime.now()
        sampler.run_mcmc(pos, cfg.sampling.nsamples, progress=True)
        time_elapsed = datetime.now() - start_time
        print(f"Time taken (hh:mm:ss.ms) to sample the posterior is : {time_elapsed}")

        # get the file name of the sampler
        fname = get_fname(cfg)

        # save the sampler
        pickle_save(sampler, "samples", fname)


def main(_):
    """
    Run the main sampling code and stores the samples.
    """
    cfg = FLAGS.config

    # run the sampler
    sample_posterior(cfg)


if __name__ == "__main__":
    app.run(main)
