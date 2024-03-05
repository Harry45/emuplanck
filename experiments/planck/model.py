"""
Code: Planck Lite likelihood code.
Date: August 2023
Author: Arrykrishna
"""

import numpy as np
import camb
from ml_collections.config_dict import ConfigDict

# Prince's code
from experiments.planck.plite import PlanckLitePy


def generate_cls(parameters: dict, cfg: ConfigDict) -> dict:
    """
    Calculate the CMB power spectra using CAMB.

    Args:
        parameters (dict): a dictionary of parameters.
        cfg (ConfigDict): the main configuration file
    Returns:
        dict: a dictionary with the power spectra and the ells.
    """
    pars = camb.CAMBparams()

    if "mnu" in cfg.cosmo.names:
        pars.set_cosmology(
            ombh2=parameters["ombh2"],
            omch2=parameters["omch2"],
            mnu=parameters["mnu"],
            omk=0,
            tau=parameters["tau"],
            thetastar=parameters["thetastar"],
            neutrino_hierarchy="normal",
        )
    else:
        pars.set_cosmology(
            ombh2=parameters["ombh2"],
            omch2=parameters["omch2"],
            omk=0,
            tau=parameters["tau"],
            thetastar=parameters["thetastar"],
        )
    pars.InitPower.set_params(As=parameters["As"], ns=parameters["ns"])
    pars.set_for_lmax(cfg.planck.ellmax, lens_potential_accuracy=cfg.planck.accuracy)
    results = camb.get_results(pars)
    powers = results.get_cmb_power_spectra(pars, CMB_unit="muK")

    # The different CL are always in the order TT, EE, BB, TE
    camb_tt = powers[cfg.planck.spectratype][:, 0]
    camb_ee = powers[cfg.planck.spectratype][:, 1]
    camb_te = powers[cfg.planck.spectratype][:, 3]

    ells = np.arange(camb_tt.shape[0])
    condition = (ells >= 2) & (ells <= 2508)

    powerspectra = {
        "ells": ells[condition],
        "tt": camb_tt[condition],
        "te": camb_te[condition],
        "ee": camb_ee[condition],
    }

    return powerspectra


def get_params(parameters: np.ndarray, cfg: ConfigDict) -> dict:
    """
    Convert an array of parameters to a dictionary of parameters.

    Args:
        parameters (np.ndarray): the parameter vector
        cfg (ConfigDict): the main configuration file
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
    if "mnu" in cfg.cosmo.names:
        params["mnu"] = parameters[6]

    return params


def calculate_loglike(
    likelihood: PlanckLitePy, points: np.ndarray, cfg: ConfigDict
) -> np.ndarray:
    """
    Calculate the log-likelihood given a set of points.

    Args:
        points (np.ndarray): a set of points of dimension d
        cfg (ConfigDict): the main configuration file

    Returns:
        np.ndarray: the log-likelihood values.
    """
    points = np.atleast_2d(points)
    npoints = points.shape[0]
    record_logl = np.zeros(npoints)
    for i in range(npoints):
        parameters = get_params(points[i], cfg)
        cls = generate_cls(parameters, cfg)
        record_logl[i] = likelihood.loglike(
            cls["tt"], cls["te"], cls["ee"], min(cls["ells"])
        )
    return record_logl
