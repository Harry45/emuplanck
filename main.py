"""
Code: Main script for sampling the Planck Lite likelihood.
Date: August 2023
Author: Arrykrishna
"""
# pylint: disable=bad-continuation
from datetime import datetime
from typing import Any, Tuple
from absl import flags, app
import numpy as np
import emcee
from ml_collections.config_flags import config_flags
from ml_collections.config_dict import ConfigDict


# our scripts and functions
from src.emulator import PlanckEmu
from src.helpers import pickle_load, pickle_save
from src.torchemu.gaussianprocess import GaussianProcess
from src.training import get_training_points, train_gp
from src.cambrun import calculate_loglike
from src.sampling import (
    generate_priors_uniform,
    generate_priors_multivariate,
    emcee_logpost,
)


FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", None, "Main configuration file.")


def get_fname(cfg: ConfigDict) -> str:
    """
    Get the file name of the sampler, depending on whether we are using:
    1) uniform prior
    2) multivariate normal prior
    3) CAMB
    4) emulator

    Args:
        cfg (ConfigDict): the main configuration file

    Returns:
        str: the file name of the sampler
    """
    if cfg.sampling.use_gp and cfg.sampling.uniform_prior:
        fname = f"samples_GP_uniform_{cfg.emu.nlhs}_{cfg.sampling.fname}"

    if cfg.sampling.use_gp and not cfg.sampling.uniform_prior:
        fname = f"samples_GP_multivariate_{cfg.emu.nlhs}_{cfg.sampling.fname}"

    if not cfg.sampling.use_gp and cfg.sampling.uniform_prior:
        fname = f"samples_CAMB_uniform_{cfg.sampling.fname}"

    if not cfg.sampling.use_gp and not cfg.sampling.uniform_prior:
        fname = f"samples_CAMB_multivariate_{cfg.sampling.fname}"
    return fname


def calculate_accuracy(cfg: ConfigDict, emulator: PlanckEmu) -> np.ndarray:
    """
    Calculate the accuracy given the predictions from the simulator and the emulator

    Args:
        cfg (ConfigDict): the main configuration file
        emulator (PlanckEmu): the emulator

    Returns:
        np.ndarray: _description_
    """
    if cfg.sampling.uniform_prior:
        priors = generate_priors_uniform(cfg)
        samples = np.column_stack(
            [priors[name].rvs(cfg.emu.ntest) for name in cfg.cosmo.names]
        )
    else:
        priors = generate_priors_multivariate(cfg)
        samples = priors.rvs(cfg.emu.ntest)

    print("Calculating accuracy")
    emu_pred = np.array(list(map(emulator.prediction, samples)))
    sim_pred = calculate_loglike(samples, cfg)
    fraction = (emu_pred - sim_pred) / sim_pred
    pickle_save(fraction, "accuracies", f"acc_{cfg.emu.nlhs}")
    return fraction


def get_priors_emulator(cfg: ConfigDict) -> Tuple[Any, GaussianProcess]:
    """
    Generate the priors and get the emulator. See config file for further details. We can
    1) generate the training points
    2) train the GP
    3) load the emulator

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
        _ = calculate_accuracy(cfg, emulator)

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
        return sampler
    return None


def main(args):
    """
    Run the main sampling code and stores the samples.
    """
    cfg = FLAGS.config

    # run the sampler
    sampler = sample_posterior(cfg)


if __name__ == "__main__":
    app.run(main)
