"""
Code: Emulator for Planck Lite likelihood code.
Date: August 2023
Author: Arrykrishna
"""

# pylint: disable=bad-continuation
import torch
import logging
import numpy as np
from ml_collections.config_dict import ConfigDict

# our script and functions
from torchemu.gaussianprocess import GaussianProcess


LOGGER = logging.getLogger(__name__)


def forward_transform(loglikelihood: np.ndarray) -> np.ndarray:
    """
    Transform the log-likelihood values such that we are emulating the log(chi2).

    Args:
        loglikelihood (np.ndarray): the log-likelihood values

    Returns:
        np.ndarray: the transformed value of the log-likelihood
    """
    ytrain = np.log(-2 * loglikelihood)
    return ytrain


def inverse_tranform(prediction: np.ndarray) -> np.ndarray:
    """
    Apply the inverse transformation on the predicted values.

    Args:
        prediction (np.ndarray): the prediction from the emulator.

    Returns:
        np.ndarray: the predicted log-likelihood value
    """
    pred_trans = -0.5 * np.exp(prediction)
    return pred_trans


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
        self.loglike = loglike
        self.inputs = inputs

        self.inputs = torch.from_numpy(inputs)
        self.ytrain = forward_transform(loglike)
        self.outputs = torch.from_numpy(self.ytrain)
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
        print(f"Training the emulator {self.cfg.emu.nrestart} times.")
        _ = self.gp_module.optimisation(
            parameters,
            niter=self.cfg.emu.niter,
            lrate=self.cfg.emu.lr,
            nrestart=self.cfg.emu.nrestart,
        )
        return self.gp_module

    def prediction(self, parameters: np.ndarray) -> float:
        """
        Predict the log-likelihood value given the pre-trained emulator.

        Args:
            parameters (np.ndarray): the test point in parameter space

        Returns:
            float: the predicted log-likelihood value
        """
        param_tensor = torch.from_numpy(parameters)
        pred_gp = self.gp_module.prediction(param_tensor)
        pred = inverse_tranform(pred_gp.item())
        return pred
