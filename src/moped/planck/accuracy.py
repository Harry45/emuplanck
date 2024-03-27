import os
import logging
import numpy as np
from ml_collections.config_dict import ConfigDict
from utils.helpers import pickle_save

# our scripts and functions
from src.emulike.planck.distribution import planck_priors_normal
from src.moped.planck.functions import PLANCKmoped, planck_moped_coefficients
from src.moped.planck.emulator import PlanckMOPEDemu
from utils.helpers import pickle_save


LOGGER = logging.getLogger(__name__)
PATH = os.path.dirname(os.path.realpath(__file__))


def planck_moped_accuracy(cfg: ConfigDict, emulators: list) -> np.ndarray:
    """
    Calculate the accuracy given the predictions from the simulator and the emulator

    Args:
        cfg (ConfigDict): the main configuration file
        emulator (list): the emulators

    Returns:
        np.ndarray: the accuracy measure
    """
    LOGGER.info("Calculating accuracy")
    model = "lcdm" if cfg.lambdacdm else "wcdm"

    # path for storing the accuracies
    path_acc = os.path.join(PATH, "accuracies")

    # generate some random points from the prior
    priors = planck_priors_normal(cfg)
    points = [priors[name].rvs(cfg.emu.ntest) for name in cfg.sampling.names]
    samples = np.column_stack(points)

    # calculate the exact MOPED coefficients
    compressor = PLANCKmoped(cfg)
    emu_pred = [
        np.array(list(map(emulators[i].prediction, samples))) for i in range(cfg.ndim)
    ]
    emu_pred = np.vstack(emu_pred).T
    sim_pred = planck_moped_coefficients(compressor, samples, cfg)

    # calculate the accuracy
    fraction = (emu_pred - sim_pred) / sim_pred

    # ignore the bad points in the loglikelihood predictions
    mask = (fraction > -100.0).all(axis=1) & (fraction < 100.0).all(axis=1)
    newfraction = fraction[mask]
    mean = np.mean(newfraction, axis=0) * 100
    std = np.std(newfraction, axis=0) * 100

    for i in range(cfg.ndim):
        LOGGER.info(f"Emulator {i} accuracy (MEAN) : {mean[i]:.2f} %")
        LOGGER.info(f"Emulator {i} accuracy (STD)  : {std[i]:.2f} %")

    pickle_save(newfraction, path_acc, f"acc_{model}_{cfg.emu.nlhs}")
    return newfraction
