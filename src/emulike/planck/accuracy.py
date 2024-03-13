import os
import logging
import numpy as np
from ml_collections.config_dict import ConfigDict
from utils.helpers import pickle_save

from src.emulike.planck.distribution import (
    planck_priors_uniform,
    planck_priors_multivariate,
)
from utils.helpers import pickle_save
from experiments.planck.plite import PlanckLitePy
from experiments.planck.model import planck_loglike
from src.emulike.planck.emulator import PlanckEmu


LOGGER = logging.getLogger(__name__)
PATH = os.path.dirname(os.path.realpath(__file__))


def calculate_planck_accuracy(cfg: ConfigDict, emulator: PlanckEmu) -> np.ndarray:
    """
    Calculate the accuracy given the predictions from the simulator and the emulator

    Args:
        cfg (ConfigDict): the main configuration file
        emulator (PlanckEmu): the emulator

    Returns:
        np.ndarray: the accuracy measure
    """
    LOGGER.info("Calculating accuracy")

    # path for storing the accuracies
    path_acc = os.path.join(PATH, "accuracies")

    # the priors
    if cfg.sampling.uniform_prior:
        priors = planck_priors_uniform(cfg)
        points = [priors[name].rvs(cfg.emu.ntest) for name in cfg.cosmo.names]
        samples = np.column_stack(points)
    else:
        priors = planck_priors_multivariate(cfg)
        samples = priors.rvs(cfg.emu.ntest)

    # the likelihood
    likelihood = PlanckLitePy(
        data_directory=cfg.path.data,
        year=cfg.planck.year,
        spectra=cfg.planck.spectra,
        use_low_ell_bins=cfg.planck.use_low_ell_bins,
    )

    # the accuracy calculation
    emu_pred = np.array(list(map(emulator.prediction, samples)))
    sim_pred = planck_loglike(likelihood, samples, cfg)
    fraction = (emu_pred - sim_pred) / sim_pred
    pickle_save(fraction, path_acc, f"acc_{cfg.emu.nlhs}")
    return fraction
