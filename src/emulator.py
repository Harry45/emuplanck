"""
Code: Emulator for Planck Lite likelihood code.
Date: August 2023
Author: Arrykrishna
"""
# pylint: disable=bad-continuation
from datetime import datetime
from typing import Any, Tuple
import torch
import numpy as np
import pandas as pd
from scipy.stats import norm, multivariate_normal
from ml_collections.config_dict import ConfigDict

# our script and functions
from src.torchemu.gaussianprocess import GaussianProcess
from src.sampling import generate_priors_uniform, generate_priors_multivariate
from src.helpers import pickle_load, pickle_save
from src.cambrun import calculate_loglike
from src.training import get_training_points, train_gp


def input_points_multivariate(
    cfg: ConfigDict, priors: multivariate_normal, nlhs: int
) -> np.ndarray:
    """
    Transform the input LH points such that they follow a multivariate normal distribution.

    Args:
        cfg (ConfigDict): the main configuration file
        priors (multivariate_normal): the multivariate normal prior.
        nlhs (int): number of LH points to use

    Returns:
        np.ndarray: the transformed LH points.
    """
    lhs_samples = pd.read_csv(f"lhs/samples_{cfg.ndim}_{nlhs}.csv", index_col=0)
    dist = norm(0, 1)
    scaled_samples = []
    for i in range(cfg.ndim):
        scaled_samples.append(dist.ppf(lhs_samples.values[:, i]))
    scaled_samples = np.column_stack(scaled_samples)
    cholesky = np.linalg.cholesky(priors.cov)
    scaled = cfg.sampling.mean.reshape(cfg.ndim, 1) + cholesky @ scaled_samples.T
    scaled = scaled.T
    return scaled


def input_points_uniform(cfg: ConfigDict, priors: dict, nlhs: int) -> np.ndarray:
    """
    Transform the input LH according to hypercube (uniform in each direction)

    Args:
        cfg (ConfigDict): the main configuration file
        priors (dict): a list of uniform priors for the cosmological parameters
        nlhs (int): the number of LH points to use

    Returns:
        np.ndarray: the scaled LH points.
    """
    lhs_samples = pd.read_csv(f"lhs/samples_{cfg.ndim}_{nlhs}.csv", index_col=0)
    scaled_samples = []
    for i, name in enumerate(cfg.cosmo.names):
        scaled_samples.append(priors[name].ppf(lhs_samples.values[:, i]))
    scaled_samples = np.column_stack(scaled_samples)
    return scaled_samples


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

    print("Calculating accuracy")
    emu_pred = np.array(list(map(emulator.prediction, samples)))
    sim_pred = calculate_loglike(samples, cfg)
    fraction = (emu_pred - sim_pred) / sim_pred
    pickle_save(fraction, "accuracies", f"acc_{cfg.emu.nlhs}")
    return fraction


def get_priors_emulator(cfg: ConfigDict) -> Tuple[Any, GaussianProcess]:
    """
    Generate the priors and get the emulator. See config file for further details. We can
    1) generate the training points
    2) train the GP
    3) load the emulator

    Args:
        cfg (ConfigDict): the main configuration file

    Returns:
        Tuple[Any, GaussianProcess]: the priors (uniform or multivariate) and the emulator
    """
    emulator = None

    if cfg.sampling.uniform_prior:
        priors = generate_priors_uniform(cfg)
        femu = f"emulator_uniform_{cfg.emu.nlhs}"
    else:
        priors = generate_priors_multivariate(cfg)
        femu = f"emulator_multivariate_{cfg.emu.nlhs}"

    if cfg.emu.generate_points:
        start_time = datetime.now()
        _ = get_training_points(cfg)
        time_elapsed = datetime.now() - start_time
        print(
            f"Time taken (hh:mm:ss.ms) to generate {cfg.emu.nlhs} training points : {time_elapsed}"
        )

    if cfg.emu.train_emu:
        start_time = datetime.now()
        emulator = train_gp(cfg)
        time_elapsed = datetime.now() - start_time
        print(
            f"Time taken (hh:mm:ss.ms) to train emulator with {cfg.emu.nlhs} training points : {time_elapsed}"
        )

    if cfg.sampling.use_gp:
        emulator = pickle_load("emulators", femu)

    if cfg.emu.calc_acc:
        _ = calculate_accuracy(cfg, emulator)

    return priors, emulator
