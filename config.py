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
    emu.nlhs = 2500
    emu.jitter = 1e-10
    emu.lr = 0.01
    emu.nrestart = 5
    emu.niter = 1000
    emu.train_emu = True
    emu.generate_points = True
    emu.calc_acc = True
    emu.ntest = 1000

    # sampling settings
    config.sampling = sampling = ConfigDict()
    sampling.run_sampler = True
    sampling.nstd = 5.0
    sampling.ncov = 2.0
    sampling.use_gp = True
    sampling.uniform_prior = False
    sampling.nsamples = 10000
    sampling.fname = "1"
    sampling.mean = np.array([0.022, 0.122, 1.041, 0.048, 3.03, 0.955])
    sampling.std = 1e-3 * np.array([0.103, 1.046, 0.219, 11.078, 22.327, 3.022])

    # public Planck 2018 chain
    # sampling.cov = 1e-8 * np.array(
    #     [
    #         [0.944, -5.238, 0.644, -9.163, -26.610, 7.736],
    #         [-5.238, 101.100, -6.971, 79.150, 405.500, -199.000],
    #         [0.644, -6.971, 5.586, -54.240, -123.100, 15.590],
    #         [-9.163, 79.150, -54.240, 6925.000, 14010.000, 273.900],
    #         [-26.610, 405.500, -123.100, 14010.000, 29260.000, -24.900],
    #         [7.736, -199.000, 15.590, 273.900, -24.900, 836.700],
    #     ]
    # )

    sampling.cov = 1e-8 * np.array(
        [
            [1.055, -6.475, 0.714, -7.161, -25.732, 12.121],
            [-6.475, 109.395, -8.132, 45.595, 353.407, -228.510],
            [0.714, -8.132, 4.795, -42.917, -101.820, 20.752],
            [-7.161, 45.595, -42.917, 12272.618, 24524.043, 386.078],
            [-25.732, 353.407, -101.820, 24524.043, 49851.000, 129.898],
            [12.121, -228.510, 20.752, 386.078, 129.898, 913.289],
        ]
    )

    return config
