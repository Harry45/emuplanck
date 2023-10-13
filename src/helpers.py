"""
Code: Some helper functions.
Date: August 2023
Author: Arrykrishna
"""
import os
import pickle
from typing import Any
from ml_collections.config_dict import ConfigDict


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

    if "mnu" in cfg.cosmo.names:
        fname += "_neutrino"
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
        pickle.dump(file, dummy)


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
