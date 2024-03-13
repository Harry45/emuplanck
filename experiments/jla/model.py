"""
Code: JLA likelihood code.
Date: March 2024
Author: Arrykrishna
"""

import numpy as np
from ml_collections.config_dict import ConfigDict

# our scripts
from experiments.jla.jlalite import JLALitePy
from experiments.jla.params import Cosmology


def get_jla_params(point: np.ndarray, cfg: ConfigDict) -> Cosmology:
    """Create a Cosmology object for the JLA likelihood.

    Args:
        point (np.ndarray): the numpy array of cosmological parameters.
        cfg (ConfigDict): the main configuration file.

    Returns:
        Cosmology: the Cosmology object.
    """
    if "w" in cfg.sampling.names:
        cosmo = Cosmology(ombh2=point[0], omch2=point[1], h=point[2], w=point[3])
    else:
        cosmo = Cosmology(ombh2=point[0], omch2=point[1], h=point[2])
    return cosmo


def jla_loglike(
    likelihood: JLALitePy, points: np.ndarray, cfg: ConfigDict
) -> np.ndarray:
    """Calculates the JLA log-likelihood over many points.

    Args:
        likelihood (JLALitePy): the JLA likelihood (already initiated).
        points (np.ndarray): an array of points (ombh2, omch2, h) for Lambda-CDM.
        cfg (ConfigDict): the main configuration file.

    Returns:
        np.ndarray: an array of the log-likelihood values.
    """
    points = np.atleast_2d(points)
    npoints = points.shape[0]
    record_logl = np.zeros(npoints)
    for i in range(npoints):
        parameters = get_jla_params(points[i], cfg)
        record_logl[i] = likelihood.loglike(parameters)
    return record_logl
