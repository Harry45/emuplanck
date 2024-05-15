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
from src.moped.ytrans import yTransformMoped
from gptemulator.gpemu import GPModel

LOGGER = logging.getLogger(__name__)


def moped_gp_models(
    cosmologies, moped_coeffs, ntrain=200, lr=0.1, noise=1e-4, verbose=True
):
    ndim = moped_coeffs.shape[1]
    models = {}
    loss = {}
    for i in range(ndim):
        ytransform = yTransformMoped(moped_coeffs[:, i])
        model = GPModel(cosmologies, ytransform)
        print(f"Training Model {i+1}")
        loss[i] = model.training(ntrain=ntrain, lr=lr, noise=noise, verbose=verbose)
        models[i] = model
    return models, loss


class JLAMOPEDemu:
    """
    Emulator for the JLA likelihood.

    Args:
        cfg (ConfigDict): the main configuration file with all the settings.
        inputs (np.ndarray): the inputs to the emulator.
        loglike (np.ndarray): the log-likelihood values.
    """

    def __init__(self, cfg: ConfigDict, inputs: np.ndarray, coeffs: np.ndarray):
        self.cfg = cfg
        self.inputs = inputs
        ytransform = yTransformMoped(coeffs)
        self.gp_module = GPModel(inputs, ytransform)

    def train_gp(self):

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
        param_tensor = torch.from_numpy(parameters)

        if self.cfg.emu.sample:
            pred_gp = self.gp_module.sample(param_tensor).item()
        else:
            pred_gp = self.gp_module.prediction(param_tensor).item()
        return pred_gp
