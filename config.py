"""
Code: The main configuration file for running the code.
Date: August 2023
Author: Arrykrishna
"""

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
    config.ndim = 6

    # cosmological parameters
    config.cosmo = cosmo = ConfigDict()
    cosmo.names = ["ombh2", "omch2", "thetastar", "tau", "As", "ns"]

    config.planck = planck = ConfigDict()
    planck.year = 2018
    planck.spectra = "TTTEEE"
    planck.use_low_ell_bins = True

    # emulator settings
    config.emu = emu = ConfigDict()
    emu.nlhs = 2000
    emu.jitter = 1e-10
    emu.lr = 0.01
    emu.nrestart = 5
    emu.niter = 1000
    emu.train_emu = False
    emu.generate_points = False
    emu.calc_acc = True
    emu.ntest = 10

    # sampling settings
    config.sampling = sampling = ConfigDict()
    sampling.run_sampler = False
    sampling.nstd = 5.0
    sampling.ncov = 2.0
    sampling.use_gp = True
    sampling.uniform_prior = False
    sampling.nsamples = 10000
    sampling.fname = "1"
    sampling.mean = np.array([0.022, 0.122, 1.041, 0.048, 3.03, 0.955])
    sampling.std = np.array([0.097, 1.006, 0.236, 8.321, 17.105, 2.892]) * 1e-3
    sampling.cov = np.array(
        [
            [9.441e-09, -5.238e-08, 6.441e-09, -9.163e-08, -2.661e-07, 7.736e-08],
            [-5.238e-08, 1.011e-06, -6.971e-08, 7.915e-07, 4.055e-06, -1.990e-06],
            [6.441e-09, -6.971e-08, 5.586e-08, -5.424e-07, -1.231e-06, 1.559e-07],
            [-9.163e-08, 7.915e-07, -5.424e-07, 6.925e-05, 1.401e-04, 2.739e-06],
            [-2.661e-07, 4.055e-06, -1.231e-06, 1.401e-04, 2.926e-04, -2.490e-07],
            [7.736e-08, -1.990e-06, 1.559e-07, 2.739e-06, -2.490e-07, 8.367e-06],
        ]
    )

    return config
