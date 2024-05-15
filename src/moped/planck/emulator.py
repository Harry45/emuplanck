"""
Code: Emulator for JLA likelihood code.
Date: March 2024
Author: Arrykrishna
"""

# pylint: disable=bad-continuation
import os
import torch
import logging
import numpy as np
import pandas as pd
from scipy.stats import norm
from ml_collections.config_dict import ConfigDict
from typing import Any

# our script and functions
from gptemulator.gpemu import GPModel
from src.moped.ytrans import yTransformMoped
from src.emulike.planck.emulator import get_mvn

LOGGER = logging.getLogger(__name__)


class PlanckMOPEDemu:
    """
    Emulator for the JLA likelihood.

    Args:
        cfg (ConfigDict): the main configuration file with all the settings.
        inputs (np.ndarray): the inputs to the emulator.
        loglike (np.ndarray): the log-likelihood values.
    """

    def __init__(self, cfg: ConfigDict, inputs: np.ndarray, loglike: np.ndarray):
        self.cfg = cfg
        ytransform = yTransformMoped(loglike)
        self.gp_module = GPModel(inputs, ytransform)
        self.mvn = get_mvn(cfg)

    def train_gp(self):
        """
        Train the Gaussian Process emulator.

        Args:
            prewhiten (bool, optional): Option to pre-whiten the input parameters. Defaults to True.

        Returns:
            GaussianProcess: the trained emulator
        """
        loss = self.gp_module.training(
            self.cfg.emu.niter,
            self.cfg.emu.lr,
            self.cfg.emu.jitter,
            self.cfg.emu.verbose,
        )
        return loss

    def prediction(self, parameters: np.ndarray) -> float:
        """
        Predict the MOPED value given the pre-trained emulator.

        Args:
            parameters (np.ndarray): the test point in parameter space

        Returns:
            float: the predicted MOPED value
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
