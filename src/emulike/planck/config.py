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

    # paths
    config.path = path = ConfigDict()
    path.parent = "/home/arrykrishna/Documents/Oxford/Projects/emuplanck/"
    path.data = os.path.join(path.parent, "experiments/planck/data")
    path.logs = os.path.join(path.parent, "src/emulike/planck/logs/")
    path.samples = os.path.join(path.parent, "src/emulike/planck/samples/")

    # cosmological parameters
    config.cosmo = cosmo = ConfigDict()
    cosmo.names = ["ombh2", "omch2", "thetastar", "tau", "As", "ns"]
    # cosmo.names = ["ombh2", "omch2", "thetastar", "tau", "As", "ns", "mnu"]
    config.ndim = len(cosmo.names)

    config.planck = planck = ConfigDict()
    planck.year = 2018
    planck.spectra = "TTTEEE"
    planck.spectratype = "total"
    planck.ellmax = 2510  # 4000
    planck.accuracy = 1  # 2
    planck.use_low_ell_bins = True

    # emulator settings
    config.emu = emu = ConfigDict()
    emu.nlhs = 2500
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
    sampling.nstd = 5.0
    sampling.ncov = 2.0
    sampling.use_gp = False
    sampling.uniform_prior = True
    sampling.nsamples = 10
    sampling.fname = "testing"

    sampling.min_uniform = np.array([0.005, 0.001, 0.5, 0.01, 2.7, 0.9])
    sampling.max_uniform = np.array([0.1, 0.99, 10.0, 0.8, 4.0, 1.1])

    sampling.mean = np.array([0.022, 0.122, 1.041, 0.048, 3.03, 0.955])
    sampling.std = 1e-3 * np.array([0.103, 1.046, 0.219, 11.078, 22.327, 3.022])

    if "mnu" in config.cosmo.names:
        sampling.min_uniform = np.append(sampling.min_uniform, 0.0)
        sampling.max_uniform = np.append(sampling.max_uniform, 0.20)

        sampling.mean = np.append(sampling.mean, 0.09)
        sampling.std = np.append(sampling.std, 6.0e-3)

    return config
