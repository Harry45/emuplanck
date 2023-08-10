"""
Code: Planck Lite likelihood code.
Date: August 2023
Author: Arrykrishna
"""

import numpy as np
import camb
from ml_collections.config_dict import ConfigDict

# Prince's code
from src.plite import PlanckLitePy


def generate_cls(parameters: dict) -> dict:
    """
    Calculate the CMB power spectra using CAMB.

    Args:
        parameters (dict): a dictionary of parameters.

    Returns:
        dict: a dictionary with the power spectra and the ells.
    """
    pars = camb.CAMBparams()
    pars.set_cosmology(
        ombh2=parameters["ombh2"],
        omch2=parameters["omch2"],
        mnu=0.06,
        omk=0,
        tau=parameters["tau"],
        thetastar=parameters["thetastar"],
    )
    pars.InitPower.set_params(As=parameters["As"], ns=parameters["ns"])
    pars.set_for_lmax(2508, lens_potential_accuracy=0)
    results = camb.get_results(pars)
    powers = results.get_cmb_power_spectra(pars, CMB_unit="muK")

    camb_tt = powers["unlensed_scalar"][:, 0]
    camb_ee = powers["unlensed_scalar"][:, 1]
    camb_te = powers["unlensed_scalar"][:, 3]

    ells = np.arange(camb_tt.shape[0])
    condition = (ells >= 2) & (ells <= 2508)

    powerspectra = {
        "ells": ells[condition],
        "tt": camb_tt[condition],
        "te": camb_te[condition],
        "ee": camb_ee[condition],
    }

    return powerspectra


def get_params(parameters: np.ndarray) -> dict:
    """
    Convert an array of parameters to a dictionary of parameters.

    Args:
        parameters (np.ndarray): the parameter vector

    Returns:
        dict: a dictionary of parameters
    """
    params = {
        "ombh2": parameters[0],
        "omch2": parameters[1],
        "thetastar": parameters[2] / 100,
        "tau": parameters[3],
        "As": np.exp(parameters[4]) * 1e-10,
        "ns": parameters[5],
    }
    return params


def calculate_loglike(points: np.ndarray, cfg: ConfigDict) -> np.ndarray:
    """
    Calculate the log-likelihood given a set of points.

    Args:
        points (np.ndarray): a set of points of dimension d
        cfg (ConfigDict): the main configuration file

    Returns:
        np.ndarray: the log-likelihood values.
    """
    likelihood = PlanckLitePy(
        data_directory="data",
        year=cfg.planck.year,
        spectra=cfg.planck.spectra,
        use_low_ell_bins=cfg.planck.use_low_ell_bins,
    )
    points = np.atleast_2d(points)
    npoints = points.shape[0]
    record_logl = np.zeros(npoints)
    for i in range(npoints):
        parameters = get_params(points[i])
        cls = generate_cls(parameters)
        record_logl[i] = likelihood.loglike(
            cls["tt"], cls["te"], cls["ee"], min(cls["ells"])
        )
    return record_logl
