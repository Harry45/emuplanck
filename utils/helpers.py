"""
Code: Some helper functions.
Date: August 2023
Author: Arrykrishna
"""

import os
import pickle
import pathlib
import logging
import numpy as np
from typing import Any
from ml_collections.config_dict import ConfigDict

LOGGER = logging.getLogger(__name__)


def get_jla_fname(cfg: ConfigDict) -> str:
    """Get the file name of the sampler for JLA:
    1) uniform prior
    2) emulator for likelihood

    Args:
        cfg (ConfigDict): the main configuration file

    Returns:
        str: the file name of the sampler
    """
    model = "lcdm" if cfg.lambdacdm else "wcdm"
    if cfg.sampling.use_gp:
        fname = f"samples_{model}_GP_{cfg.emu.nlhs}_{cfg.sampling.fname}"
    else:
        fname = f"samples_{model}_Analytic_{cfg.sampling.fname}"
    return fname


def get_planck_fname(cfg: ConfigDict) -> str:
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
    model = "lcdm" if cfg.lambdacdm else "wcdm"
    if cfg.sampling.use_gp:
        fname = f"samples_{model}_GP_{cfg.emu.nlhs}_{cfg.sampling.fname}"
    else:
        fname = f"samples_{model}_CAMB_{cfg.sampling.fname}"
    return fname


def get_jla_planck_fname(cfg_jla: ConfigDict, cfg_planck: ConfigDict) -> str:
    """
    Get the filename of the sampler.

    Args:
        cfg_jla (ConfigDict): the main configuration file for JLA
        cfg_planck (ConfigDict): the main configuration file for Planck

    Returns:
        str: the file name of the sampler
    """
    model = "lcdm" if cfg_planck.lambdacdm else "wcdm"
    if cfg_planck.sampling.use_gp and cfg_jla.sampling.use_gp:
        fname = f"samples_{model}_emulator_{cfg_planck.sampling.fname}"
    else:
        fname = f"samples_{model}_simulator_{cfg_planck.sampling.fname}"
    return fname


def pickle_save(file: list, folder: str, fname: str) -> None:
    """Stores a list in a folder.
    Args:
        list_to_store (list): The list to store.
        folder_name (str): The name of the folder.
        file_name (str): The name of the file.
    """

    # create the folder if it does not exist
    os.makedirs(folder, exist_ok=True)

    # use compressed format to store data
    path = os.path.join(folder, fname)
    with open(path + ".pkl", "wb") as dummy:
        pickle.dump(file, dummy, protocol=pickle.HIGHEST_PROTOCOL)


def pickle_load(folder: str, fname: str) -> Any:
    """Reads a list from a folder.
    Args:
        folder_name (str): The name of the folder.
        file_name (str): The name of the file.
    Returns:
        Any: the stored file
    """
    path = os.path.join(folder, fname)
    with open(path + ".pkl", "rb") as dummy:
        file = pickle.load(dummy)
    return file


def process_chains(cfg: ConfigDict, fname: str, thin: int, discard: int) -> np.ndarray:
    """Process the MCMC chain (EMCEE object) such that we discard the first
    N samples and take every i sample. We also store the mean and covariance
    of the final chain.

    Args:
        cfg (ConfigDict): the main configuration file
        fname (str): name of the file
        thin (int): thinning factor
        discard (int): number of samples to discard

    Returns:
        np.ndarray: the processed chain
    """
    path, file = os.path.split(fname)
    fullpath = os.path.join(cfg.path.parent, path)
    emcee_file = pickle_load(fullpath, file)
    emcee_samples = emcee_file.get_chain(flat=True, thin=thin, discard=discard)
    mean_samples = np.mean(emcee_samples, 0)
    cov_samples = np.cov(emcee_samples.T)

    LOGGER.info(f"Number of samples after processing : {emcee_samples.shape[0]}")

    pickle_save(emcee_samples, fullpath, file + "_thinned")
    pickle_save(mean_samples, fullpath, file + "_mean")
    pickle_save(cov_samples, fullpath, file + "_cov")
    return emcee_samples


def emcee_chains(cfg: ConfigDict, fname: str, thin: int, discard: int) -> np.ndarray:
    """Generate the MCMC chain for running MCEvidence.

    Args:
        cfg (ConfigDict): the main configuration file
        fname (str): the full path to the file, for example:

        fname = "src/emulike/planck/samples/samples_lcdm_GP_1500_experiment_1"

        thin (int): thinning factor
        discard (int): number of samples to discard

    Returns:
        np.ndarray: the final array to be used for running MCEvidence.
    """
    # the pkl file
    path, file = os.path.split(fname)
    fullpath = os.path.join(cfg.path.parent, path)
    pkl_file = pickle_load(cfg.path.parent, fname)

    # the MCMC samples
    array = pkl_file.get_chain(flat=True, thin=thin, discard=discard)

    # number of samples
    nsamples = array.shape[0]

    # the log-posterior
    logp = pkl_file.get_log_prob(flat=True, thin=thin, discard=discard).reshape(
        nsamples, 1
    )

    # the samples are unique - so we add a column of ones in the beginning
    ones = np.ones((nsamples, 1))

    # combine the important information
    comb = np.concatenate([ones, logp, array], axis=1)

    # save the array if required
    np.savetxt(fullpath + "/" + "MCE_" + file, comb)
    return comb
