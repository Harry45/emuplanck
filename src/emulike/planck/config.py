"""
Code: The main configuration file for running the code.
Date: August 2023
Author: Arrykrishna
"""

import os
import numpy as np
from ml_collections.config_dict import ConfigDict


def get_config(experiment: str) -> ConfigDict:
    """
    Get the main configuration file

    Args:
        experiment (str): the experiment we want to run

    Returns:
        ConfigDict: the main configuration file
    """
    config = ConfigDict()
    config.logname = experiment
    config.experiment = experiment
    config.lambdacdm = True

    # paths
    config.path = path = ConfigDict()
    # path.parent = "/home/arrykrishna/Documents/Oxford/Projects/emuplanck/"
    path.parent = "/mnt/users/phys2286/projects/emuplanck"
    path.data = os.path.join(path.parent, "experiments/planck/data")

    config.planck = planck = ConfigDict()
    planck.year = 2018
    planck.spectra = "TTTEEE"
    planck.spectratype = "total"
    planck.ellmax = 4000
    planck.accuracy = 2
    planck.use_low_ell_bins = True

    # emulator settings
    config.emu = emu = ConfigDict()
    emu.nlhs = 1500
    emu.jitter = 1e-10
    emu.lr = 0.01
    emu.nrestart = 5
    emu.niter = 1000
    emu.train_emu = False
    emu.generate_points = False
    emu.calc_acc = False
    emu.ntest = 1000

    # sampling settings
    config.sampling = sampling = ConfigDict()
    sampling.run_sampler = True
    sampling.use_gp = False
    sampling.nsamples = 5
    sampling.fname = "testing"
    sampling.mean = np.array([0.022, 0.12, 0.7, 3.05, 0.965])
    # sampling.std = np.array([2e-4, 2e-3, 1e-2, 2e-2, 1e-2])
    sampling.std = np.array([1e-3, 0.025, 0.05, 0.05, 0.025])

    if config.lambdacdm:
        sampling.names = ["ombh2", "omch2", "h", "As", "ns"]
    else:
        sampling.names = ["ombh2", "omch2", "h", "As", "ns", "w"]
        sampling.mean = np.append(sampling.mean, -1.0)
        sampling.std = np.append(sampling.std, 0.1)

    # number of dimensions
    config.ndim = len(sampling.names)

    return config
