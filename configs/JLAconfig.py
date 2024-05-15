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
    config.ncovnuisance = 5

    # paths
    config.path = path = ConfigDict()
    path.parent = "/home/arrykrishna/Documents/Oxford/Projects/emuplanck/"
    # path.parent = "/mnt/users/phys2286/projects/emuplanck"
    path.data = os.path.join(path.parent, "experiments/jla/data")

    # optimisation (we keep ombh2 fixed to 0.019)
    config.opt = opt = ConfigDict()
    if config.lambdacdm:
        opt.sol_name = "solution_lambda"
        opt.cov_name = "covariance_lambda"
        opt.names = ["omch2", "h", "Mb", "delta_M", "alpha", "beta"]
        opt.initial = [0.12, 0.7, -19.0, 0.0, 0.125, 2.60]
    else:
        opt.sol_name = "solution_w"
        opt.cov_name = "covariance_w"
        opt.names = ["omch2", "h", "w", "Mb", "delta_M", "alpha", "beta"]
        opt.initial = [0.12, 0.7, -1.0, -19.0, 0.0, 0.125, 2.60]

    # emulator settings
    config.emu = emu = ConfigDict()
    emu.nlhs = 700
    emu.jitter = 1e-4
    emu.lr = 0.3
    emu.niter = 200
    emu.verbose = True
    emu.sample = False

    emu.train_emu = True
    emu.generate_points = False
    emu.calc_acc = True
    emu.ntest = 1000

    # cosmological parameters
    config.sampling = sampling = ConfigDict()
    sampling.use_gp = True
    sampling.run_sampler = True
    sampling.fname = "test"
    sampling.nsamples = 1
    sampling.mean = np.array([0.022, 0.12, 0.67])
    sampling.std = np.array([1e-3, 0.025, 0.05])

    # compression
    config.moped = moped = ConfigDict()
    moped.eps = [1e-3, 1e-2, 1e-2]

    if config.lambdacdm:
        sampling.names = ["ombh2", "omch2", "h"]
    else:
        sampling.names = ["ombh2", "omch2", "h", "w"]
        sampling.mean = np.append(sampling.mean, -1.0)
        sampling.std = np.append(sampling.std, 0.1)
        moped.eps = moped.eps.append(1e-2)

    # number of dimensions
    config.ndim = len(sampling.names)
    return config
