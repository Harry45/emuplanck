"""
Code: Emulator for Planck Lite likelihood code.
Date: August 2023
Author: Arrykrishna
"""

# pylint: disable=bad-continuation
import os
import torch
import logging
from typing import Any
import numpy as np
from scipy.stats import multivariate_normal
from ml_collections.config_dict import ConfigDict

# our script and functions
from gptemulator.gpemu import GPModel
from src.emulike.ytrans import yTransformLogLike
from utils.helpers import pickle_load

LOGGER = logging.getLogger(__name__)


def get_mvn(cfg: ConfigDict) -> Any:
    """Generates a multivariate normal distribution given the simulator runs.

    Args:
        cfg (ConfigDict): the main configuration file.

    Returns:
        Any: the multivariate normal distribution
    """
    path, file = os.path.split(cfg.emu.sim_path)
    fullpath = os.path.join(cfg.path.parent, path)
    mean = pickle_load(fullpath, file + "_mean")
    cov = pickle_load(fullpath, file + "_cov")
    return multivariate_normal(mean, cfg.emu.ncov * cov)


class PlanckEmu:
    """
    Emulator for the Planck likelihood.

    Args:
        cfg (ConfigDict): the main configuration file with all the settings.
        inputs (np.ndarray): the inputs to the emulator.
        loglike (np.ndarray): the log-likelihood values.
    """

    def __init__(self, cfg: ConfigDict, inputs: np.ndarray, loglike: np.ndarray):
        self.cfg = cfg
        ytransform = yTransformLogLike(loglike)
        self.mvn = get_mvn(cfg)
        self.gp_module = GPModel(inputs, ytransform)

    def train_gp(self):
        LOGGER.info(f"Training the likelihood emulator")
        loss = self.gp_module.training(
            self.cfg.emu.niter,
            self.cfg.emu.lr,
            self.cfg.emu.jitter,
            self.cfg.emu.verbose,
        )
        return loss

    def prediction(self, parameters: np.ndarray) -> float:
        """
        Predict the log-likelihood value given the pre-trained emulator.

        Args:
            parameters (np.ndarray): the test point in parameter space

        Returns:
            float: the predicted log-likelihood value
        """
        pdf = self.mvn.pdf(parameters)
        param_tensor = torch.from_numpy(parameters)

        if pdf > 1e-3:
            if self.cfg.emu.sample:
                pred_gp = self.gp_module.sample(param_tensor).item()
            else:
                pred_gp = self.gp_module.prediction(param_tensor).item()
            return pred_gp
        return -1e32
