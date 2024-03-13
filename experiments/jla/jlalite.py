"""
Code: JLA theory calculation.
Date: March 2024
Author: Arrykrishna
"""

import glob
import logging
from astropy.io import fits
from typing import Tuple
import numpy as np
import os
from scipy.optimize import minimize
import pandas as pd
from ml_collections.config_dict import ConfigDict


# our files
from utils.helpers import pickle_save, pickle_load
from experiments.jla.params import Cosmology, Nuisance

LOGGER = logging.getLogger(__name__)
PATH = os.path.dirname(os.path.realpath(__file__))
CLIGHT = 299792.458
NREDSHIFT = 1000


def jla_integration(cosmo: Cosmology, redshift: float) -> np.ndarray:
    """The cosmological function to integrate.

    Args:
        cosmo (Cosmology): Cosmology object containing the parameters
        redshift (float): the value of the redshift

    Returns:
        np.ndarray: the value of the cosmological function
    """

    function = np.sqrt(
        cosmo.Omega_m * (1.0 + redshift) ** 3.0
        + (1.0 - cosmo.Omega_m) * (1.0 + redshift) ** (3.0 * (1.0 + cosmo.w))
    )

    return 1.0 / function


def jla_cosmo_model(cosmo: Cosmology, redshift: float) -> np.ndarray:
    """Calculate the luminosity distance at a given redshift and cosmological parameters.

    Args:
        cosmo (Cosmology): Cosmology object containing the parameters
        redshift (float): the value of the redshift

    Returns:
        torch.tensor: the luminosity distance
    """

    # multiplying factor
    factor = 10**3 * (1.0 + redshift) * CLIGHT / cosmo.h

    # so we use trapezoidal rule - need a redshift grid first
    zgrid = np.linspace(0.0, redshift, NREDSHIFT)

    # the function to integrate
    fgrid = jla_integration(cosmo, zgrid)

    # the integral
    int_val = 5.0 * np.log10(factor * np.trapz(fgrid, zgrid))

    return int_val


def jla_nuisance_model(
    nuisance: Nuisance, log_stellar: float, x1: float, color: float
) -> np.ndarray:
    """Calculates the theoretical model corresponding to the nuisance parameters.

    Args:
        nuisance (Nuisance): object of nuisance parameters
        log_stellar (float): the log-stellar mass
        x1 (float): the x1 value
        color (float): the color value

    Returns:
        np.ndarray: the nuisance model
    """

    if log_stellar >= 10.0:
        dummy = 1.0
    else:
        dummy = 0.0

    nuisa_model = (
        nuisance.Mb
        + dummy * nuisance.delta_M
        - x1 * nuisance.alpha
        + color * nuisance.beta
    )

    return nuisa_model


def cosmo_nuisance_model(
    cosmo: Cosmology,
    nuisance: Nuisance,
    redshifts: np.ndarray,
    log_stellar: np.ndarray,
    x1: np.ndarray,
    color: np.ndarray,
) -> np.ndarray:
    """Calculates the theoretical model (cosmology and nuisance)

    Args:
        cosmo (Cosmology): Cosmology object containing the parameters
        nuisance (Nuisance): object of nuisance parameters
        redshifts (np.ndarray): the redshifts of the supernova
        log_stellar (np.ndarray): the log stellar mass
        x1 (np.ndarray): x1 values from the catalog
        color (np.ndarray): the color values from the catalog

    Returns:
        np.ndarray: the full theoretical model
    """
    ndata = len(redshifts)
    theory = np.zeros(ndata)
    for i in range(ndata):
        cosmo_part = jla_cosmo_model(cosmo, redshifts[i])
        nuisance_part = jla_nuisance_model(nuisance, log_stellar[i], x1[i], color[i])
        theory[i] = cosmo_part + nuisance_part
    return theory


def theory_covariance(sigma_mu: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    """Computes the theoretical covariance matrix according to the explanation in the Betoule et al. paper.

    Args:
        sigma_mu (np.ndarray): error in mu from the catalog
        alpha (float): the alpha parameter
        beta (float): the beta parameter

    Returns:
        np.ndarray: the new covariance matrix
    """
    covpath = os.path.join(PATH, "data/C*.fits")
    cov_eta = sum([fits.open(mat)[0].data for mat in glob.glob(covpath)])

    # number of data points
    ndata = cov_eta.shape[0] // 3

    # the covariance matrix
    cov = np.zeros((ndata, ndata))

    for i, coef1 in enumerate([1.0, alpha, -beta]):
        for j, coef2 in enumerate([1.0, alpha, -beta]):
            cov += (coef1 * coef2) * cov_eta[i::3, j::3]

    # Add diagonal term from Eq. 13
    sigma_pecvel = (5 * 150 / 3e5) / (np.log(10.0) * sigma_mu[:, 2])
    cov += np.eye(ndata) * (sigma_mu[:, 0] ** 2 + sigma_mu[:, 1] ** 2 + sigma_pecvel**2)

    return cov


def loss_function(theta: np.ndarray, inputs: dict, cfg: ConfigDict) -> float:
    """Calculates the loss function for a given point.

    Args:
        theta (np.ndarray): the vector of parameters
        inputs (dict): the input dictionary with all the information
        cfg (ConfigDict): the main configuration file

    Returns:
        float: the loss
    """

    # ombh2 is fixed to 0.019
    if cfg.lambdacdm:
        cosmo = Cosmology(omch2=theta[0], h=theta[1])
        nuisance = Nuisance(
            Mb=theta[2], delta_M=theta[3], alpha=theta[4], beta=theta[5]
        )
    else:
        cosmo = Cosmology(omch2=theta[0], h=theta[1], w=theta[2])
        nuisance = Nuisance(
            Mb=theta[3], delta_M=theta[4], alpha=theta[5], beta=theta[6]
        )

    # get the updated covariance matrix
    covariance = theory_covariance(inputs["smu"], nuisance.alpha, nuisance.beta)
    input_data = (inputs["z"], inputs["ls"], inputs["x1"], inputs["c"])
    model = cosmo_nuisance_model(cosmo, nuisance, *input_data)

    # calculate the loss
    diff = model - inputs["mb"]
    loss = diff @ np.linalg.solve(covariance, diff) + np.linalg.slogdet(covariance)[1]
    return loss


def optimisation(
    initial: np.ndarray, inputs: dict, cfg: ConfigDict, save: bool = True, **kwargs
):
    """Optimise for the parameters in the model

    Args:
        initial (np.ndarray): the initial point of the optimiser
        inputs (dict): a dictionary with keys:
            smu : sigma_mu
            z   : redshifts
            ls  : log stellar mass
            x1  : x1
            c   : color
            mb  : data
        cfg (ConfigDict): the main configuration file
        save (bool, optional): Save the optimal points. Defaults to True.
    """

    soln = minimize(loss_function, initial, args=(inputs, cfg), **kwargs)
    opt = soln.x
    if cfg.lambdacdm:
        nuisance = Nuisance(Mb=opt[2], delta_M=opt[3], alpha=opt[4], beta=opt[5])
    else:
        nuisance = Nuisance(Mb=opt[3], delta_M=opt[4], alpha=opt[5], beta=opt[6])

    covariance = theory_covariance(inputs["smu"], nuisance.alpha, nuisance.beta)
    soln_path = os.path.join(PATH, "optimal")
    if save:
        pickle_save(soln, soln_path, cfg.opt.sol_name)
        pickle_save(covariance, soln_path, cfg.opt.cov_name)
    return soln, covariance


def check_optimisation(inputs: dict, cfg: ConfigDict) -> Tuple[np.ndarray, np.ndarray]:
    """Check if we have the optimised solutions.

    Args:
        inputs (dict): the inputs with all the key information
        cfg (ConfigDict): the main configuration file

    Returns:
        Tuple[np.ndarray, np.ndarray]: the solution and the covariance
    """
    soln_path = os.path.join(PATH, f"optimal/{cfg.opt.sol_name}.pkl")
    cov_path = os.path.join(PATH, f"optimal/{cfg.opt.cov_name}.pkl")
    mle_exists = os.path.isfile(soln_path)
    cov_exists = os.path.isfile(cov_path)
    if mle_exists and cov_exists:
        soln = pickle_load(PATH, f"optimal/{cfg.opt.sol_name}")
        cov = pickle_load(PATH, f"optimal/{cfg.opt.cov_name}")
        return soln, cov
    LOGGER.info("Cannot find MLE and covariance: Running optimisation.")
    soln, cov = optimisation(cfg.opt.initial, inputs, cfg, save=True)
    return soln, cov


def marginalise_nuisance(
    inputs: dict, cfg: ConfigDict
) -> Tuple[np.ndarray, np.ndarray]:
    """Marginalise over the nuisance parameters

    Args:
        inputs (dict): the input dictionary with all the key information
        cfg (ConfigDict): the main configuration file

    Returns:
        Tuple[np.ndarray, np.ndarray]: the new data and new covariance
    """
    LOGGER.info("Quick marginalising over nuisance parameters")
    ndata = inputs["mb"].shape[0]
    psi = np.zeros((ndata, 4))
    psi[:, 0] = 1.0
    psi[:, 1][inputs["ls"] >= 10.0] = 1.0
    psi[:, 2] = -inputs["x1"]
    psi[:, 3] = inputs["c"]
    soln, cov = check_optimisation(inputs, cfg)
    mean = soln.x[-4:]
    newdata = inputs["mb"] - psi @ mean
    prior_cov = cfg.ncovnuisance * soln.hess_inv[-4:, -4:]
    newcov = cov + psi @ prior_cov @ psi.T
    return newdata, newcov


class JLALitePy:
    def __init__(self, cfg: ConfigDict):
        self.inputs = self.load_data()
        self.newdata, self.newcov = marginalise_nuisance(self.inputs, cfg)

    def load_data(self) -> dict:
        """Load all data for computing the likelihood. The dictionary consists of the following keys:
        mb  : the data
        smu : sigma_mu
        z   : redshifts
        ls  : log stellar mass
        x1  : x1
        c   : color
        mb  : data

        Returns:
            dict: the dictionary with the above keys and values
        """
        data_path = os.path.join(PATH, "data")
        lc_params_path = os.path.join(data_path, "jla_lcparams.txt")
        sigma_mu_path = os.path.join(data_path, "sigma_mu.txt")
        light_curve = pd.read_csv(lc_params_path, sep=" ", header=0)
        sigma_mu = np.loadtxt(sigma_mu_path)

        # quantities we need
        log_stellar_mass = light_curve["3rdvar"].values
        x1 = light_curve["x1"].values
        color = light_curve["color"].values
        redshifts = light_curve["zcmb"].values
        data = light_curve["mb"].values
        inputs = {
            "mb": data,
            "ls": log_stellar_mass,
            "x1": x1,
            "c": color,
            "z": redshifts,
            "smu": sigma_mu,
        }
        return inputs

    def theory(self, cosmo: Cosmology) -> np.ndarray:
        """Calculate the theory of the cosmological model.

        Args:
            cosmo (Cosmology): the cosmology object

        Returns:
            np.ndarray: the theoretical values
        """
        ndata = len(self.newdata)
        values = np.zeros(ndata)
        for i in range(ndata):
            values[i] = jla_cosmo_model(cosmo, self.inputs["z"][i])
        return values

    def loglike(self, cosmo: Cosmology) -> float:
        """Calculate the log-likelihood.

        Args:
            cosmo (Cosmology): the cosmology object.

        Returns:
            float: the log-likelihood value
        """
        diff = self.theory(cosmo) - self.newdata
        return -0.5 * diff @ np.linalg.solve(self.newcov, diff)
