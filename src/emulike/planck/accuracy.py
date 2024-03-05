import logging
import numpy as np
from ml_collections.config_dict import ConfigDict
from utils.helpers import pickle_save

from src.emulike.planck.distribution import (
    generate_priors_uniform,
    generate_priors_multivariate,
)
from utils.helpers import pickle_save
from experiments.planck.model import calculate_loglike
from src.emulike.planck.emulator import PlanckEmu

LOGGER = logging.getLogger(__name__)


def calculate_accuracy(cfg: ConfigDict, emulator: PlanckEmu) -> np.ndarray:
    """
    Calculate the accuracy given the predictions from the simulator and the emulator

    Args:
        cfg (ConfigDict): the main configuration file
        emulator (PlanckEmu): the emulator

    Returns:
        np.ndarray: the accuracy measure
    """
    if cfg.sampling.uniform_prior:
        priors = generate_priors_uniform(cfg)
        samples = np.column_stack(
            [priors[name].rvs(cfg.emu.ntest) for name in cfg.cosmo.names]
        )
    else:
        priors = generate_priors_multivariate(cfg)
        samples = priors.rvs(cfg.emu.ntest)

    LOGGER.info("Calculating accuracy")
    emu_pred = np.array(list(map(emulator.prediction, samples)))
    sim_pred = calculate_loglike(samples, cfg)
    fraction = (emu_pred - sim_pred) / sim_pred
    pickle_save(fraction, "accuracies", f"acc_{cfg.emu.nlhs}")
    return fraction
