import logging
import numpy as np
from ml_collections.config_dict import ConfigDict
from experiments.planck.model import planck_get_params, planck_theory
from experiments.planck.params import PCosmology
from experiments.planck.plite import PlanckLitePy
from src.moped.compression import vectors, MOPEDstore

LOGGER = logging.getLogger(__name__)


def planck_theory_vector(
    likelihood: PlanckLitePy, point: PCosmology, cfg: ConfigDict
) -> np.ndarray:
    """
    Calculate the Planck theory vector given a set of points.

    Args:
        point (PCosmology): a set of points of dimension d
        cfg (ConfigDict): the main configuration file

    Returns:
        np.ndarray: the theory vector.
    """
    cls = planck_theory(point, cfg)
    theory = likelihood.theory(cls["tt"], cls["te"], cls["ee"], min(cls["ells"]))
    return theory


def planck_gradient_theory(
    likelihood: PlanckLitePy,
    cfg: ConfigDict,
    expansion: np.ndarray = None,
) -> np.ndarray:
    """Calculates the gradient of the Planck theory with respect to the input parameters.

    Args:
        likelihood (PlanckLitePy): the Planck likelihood.
        cfg (ConfigDict): the main configuration file.
        expansion (np.ndarray, optional): An optional expansion point. Defaults to None.

    Returns:
        np.ndarray: the first derivatives of size N x p.
    """

    if expansion is None:
        expansion = cfg.sampling.mean
    ndim = len(expansion)
    gradient_theory = []
    for i in range(ndim):
        p_plus = np.copy(expansion)
        p_minus = np.copy(expansion)
        p_plus[i] = p_plus[i] + cfg.moped.eps[i]
        p_minus[i] = p_minus[i] - cfg.moped.eps[i]
        LOGGER.info(f"Theory derivative is calculated at : {p_plus}")
        LOGGER.info(f"Theory derivative is calculated at : {p_minus}")
        p_plus = planck_get_params(p_plus, cfg)
        p_minus = planck_get_params(p_minus, cfg)
        theory_plus = planck_theory_vector(likelihood, p_plus, cfg)
        theory_minus = planck_theory_vector(likelihood, p_minus, cfg)
        gradient_theory.append((theory_plus - theory_minus) / (2.0 * cfg.moped.eps[i]))
    return np.vstack(gradient_theory).T


class PLANCKmoped:
    def __init__(self, cfg: ConfigDict, expansion: np.ndarray = None):
        """Calculates the compressed data/theory vector using MOPED. The expansion point should be
        a numpy array with values corresponding to:

        cosmo = ["ombh2", "omch2", "h", "As", "ns"]

        Args:
            cfg (ConfigDict): the main configuration file
            expansion (np.ndarray, optional): an optional expansion point. Defaults to None.
        """
        self.likelihood = PlanckLitePy(
            data_directory=cfg.path.data,
            year=cfg.planck.year,
            spectra=cfg.planck.spectra,
            use_low_ell_bins=cfg.planck.use_low_ell_bins,
        )
        self.expansion = expansion
        self.cfg = cfg
        self.store = self._postinit()

    def _postinit(self) -> MOPEDstore:
        """Calculates the B matrix and the compressed data y. See

        https://arxiv.org/abs/astro-ph/9911102

        Returns:
            MOPEDstore: Store consisting of b_matrix and ycomp.
        """
        grad = planck_gradient_theory(self.likelihood, self.cfg, self.expansion)

        # Prince stores the Fisher matrix
        covariance = np.linalg.inv(self.likelihood.fisher)
        moped_vectors = vectors(grad, covariance)
        ycomp = moped_vectors.T @ self.likelihood.X_data
        return MOPEDstore(b_matrix=moped_vectors, ycomp=ycomp)

    def compression(self, point: PCosmology) -> np.ndarray:
        """Given a test point in parameter space, we compress this to p numbers.

        Args:
            point (PCosmology): the input test point.

        Returns:
            np.ndarray: the compressed theory at that point.
        """
        theory = planck_theory_vector(self.likelihood, point, self.cfg)
        return self.store.b_matrix.T @ theory


def planck_moped_coefficients(
    compressor: PLANCKmoped, points: np.ndarray, cfg: ConfigDict
) -> np.ndarray:
    """Given the compressor, we will calculate the MOPED coefficients for each
    input parameter.

    Args:
        compressor (PLANCKmoped): the Planck compressor.
        points (np.ndarray): the input trraining points.
        cfg (ConfigDict): the main configuration file

    Returns:
        np.ndarray: the MOPED coefficients
    """
    points = np.atleast_2d(points)
    npoints, ndim = points.shape
    record_coeff = np.zeros((npoints, ndim))
    for i in range(npoints):
        parameters = planck_get_params(points[i], cfg)
        record_coeff[i] = compressor.compression(parameters)
    return record_coeff
