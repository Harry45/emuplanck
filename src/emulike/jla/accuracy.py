import os
import logging
import numpy as np
from ml_collections.config_dict import ConfigDict
from utils.helpers import pickle_save

from experiments.jla.jlalite import JLALitePy
from experiments.jla.model import jla_loglike
from src.emulike.jla.emulator import JLAemu
from src.emulike.jla.distribution import jla_priors_normal
from utils.helpers import pickle_save


LOGGER = logging.getLogger(__name__)
PATH = os.path.dirname(os.path.realpath(__file__))


def calculate_jla_accuracy(cfg: ConfigDict, emulator: JLAemu) -> np.ndarray:
    """
    Calculate the accuracy given the predictions from the simulator and the emulator

    Args:
        cfg (ConfigDict): the main configuration file
        emulator (PlanckEmu): the emulator

    Returns:
        np.ndarray: the accuracy measure
    """
    LOGGER.info("Calculating accuracy")
    model = "lcdm" if cfg.lambdacdm else "wcdm"
    # path for storing the accuracies
    path_acc = os.path.join(PATH, "accuracies")

    # generate some random points from the prior
    priors = jla_priors_normal(cfg)
    points = [priors[name].rvs(cfg.emu.ntest) for name in cfg.sampling.names]
    samples = np.column_stack(points)

    # calculate the exact likelihood
    likelihood = JLALitePy(cfg)
    emu_pred = np.array(list(map(emulator.prediction, samples)))
    sim_pred = jla_loglike(likelihood, samples, cfg)

    # calculate the accuracy
    fraction = (emu_pred - sim_pred) / sim_pred

    # ignore the bad points in the loglikelihood predictions
    newfraction = fraction[fraction < 100.0]
    pickle_save(newfraction, path_acc, f"acc_{model}_{cfg.emu.nlhs}")
    LOGGER.info(f"Emulator accuracy: {np.mean(newfraction)*100:.2f} %")
    return fraction
