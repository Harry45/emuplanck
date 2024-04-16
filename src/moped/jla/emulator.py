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
from torchemu.gaussianprocess import GaussianProcess


LOGGER = logging.getLogger(__name__)


def forward_transform(value: np.ndarray) -> np.ndarray:
    """
    Implement a forward transformation if we want to.

    Args:
        value (np.ndarray): the log-likelihood value or MOPED coefficient

    Returns:
        np.ndarray: the transformed value of the log-likelihood
    """
    ytrain = value
    return ytrain


def inverse_tranform(prediction: np.ndarray) -> np.ndarray:
    """
    Apply the inverse transformation on the predicted values.

    Args:
        prediction (np.ndarray): the prediction from the emulator.

    Returns:
        np.ndarray: the predicted log-likelihood value
    """
    pred_trans = prediction
    return pred_trans


class JLAMOPEDemu:
    """
    Emulator for the JLA likelihood.

    Args:
        cfg (ConfigDict): the main configuration file with all the settings.
        inputs (np.ndarray): the inputs to the emulator.
        loglike (np.ndarray): the log-likelihood values.
    """

    def __init__(self, cfg: ConfigDict, inputs: np.ndarray, loglike: np.ndarray):
        self.cfg = cfg
        self.loglike = loglike
        self.inputs = inputs

        self.inputs = torch.from_numpy(inputs)
        ytrans = forward_transform(loglike)
        self.ymean = np.mean(ytrans)
        self.ystd = np.std(ytrans)
        ytrain = (ytrans - self.ymean) / self.ystd
        self.outputs = torch.from_numpy(ytrain)
        self.gp_module = None

    def train_gp(self, prewhiten: bool = True) -> GaussianProcess:
        """
        Train the Gaussian Process emulator.

        Args:
            prewhiten (bool, optional): Option to pre-whiten the input parameters. Defaults to True.

        Returns:
            GaussianProcess: the trained emulator
        """

        self.gp_module = GaussianProcess(self.cfg, self.inputs, self.outputs, prewhiten)
        parameters = torch.randn(self.cfg.ndim + 1)
        LOGGER.info(f"Training MOPED emulator {self.cfg.emu.nrestart} times.")
        _ = self.gp_module.optimisation(
            parameters,
            niter=self.cfg.emu.niter,
            lrate=self.cfg.emu.lr,
            nrestart=self.cfg.emu.nrestart,
        )
        return self.gp_module

    def prediction(self, parameters: np.ndarray) -> float:
        """
        Predict the MOPED value given the pre-trained emulator.

        Args:
            parameters (np.ndarray): the test point in parameter space

        Returns:
            float: the predicted MOPED value
        """
        param_tensor = torch.from_numpy(parameters)
        pred_gp = self.gp_module.prediction(param_tensor).item()

        # prediction must be within limits of standard normal
        # we consider 6 sigma limit
        if -6.0 <= pred_gp <= 6.0:
            pred = inverse_tranform(self.ystd * pred_gp + self.ymean)
            return pred
        return -1e32
