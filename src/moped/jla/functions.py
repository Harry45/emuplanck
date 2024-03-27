import logging
import numpy as np
from ml_collections.config_dict import ConfigDict


from experiments.jla.jlalite import JLALitePy
from experiments.jla.params import Cosmology
from experiments.jla.model import jla_loglike, get_jla_params
from src.moped.compression import vectors, MOPEDstore

LOGGER = logging.getLogger(__name__)


def jla_gradient_theory(
    likelihood: JLALitePy,
    cfg: ConfigDict,
    expansion: np.ndarray = None,
) -> np.ndarray:
    """Calculates the gradient of the JLA theory with respect to the input parameters.

    Args:
        likelihood (JLALitePy): the JLA likelihood.
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
        p_plus = get_jla_params(p_plus, cfg)
        p_minus = get_jla_params(p_minus, cfg)
        theory_plus = likelihood.theory(p_plus)
        theory_minus = likelihood.theory(p_minus)
        gradient_theory.append((theory_plus - theory_minus) / (2.0 * cfg.moped.eps[i]))
    return np.vstack(gradient_theory).T


class JLAmoped:
    """Calculates the compressed data/theory vector using MOPED. The expansion point should be
    a numpy array with values corresponding to:

    cosmo = Cosmology(ombh2=point[0], omch2=point[1], h=point[2])

    Args:
        cfg (ConfigDict): the main configuration file
        expansion (np.ndarray, optional): an optional expansion point. Defaults to None.
    """

    def __init__(self, cfg: ConfigDict, expansion: np.ndarray = None) -> MOPEDstore:

        self.likelihood = JLALitePy(cfg)
        self.expansion = expansion
        self.cfg = cfg
        self.store = self._postinit()

    def _postinit(self) -> MOPEDstore:
        """Calculates the B matrix and the compressed data y. See

        https://arxiv.org/abs/astro-ph/9911102

        Returns:
            MOPEDstore: Store consisting of b_matrix and ycomp.
        """
        grad = jla_gradient_theory(self.likelihood, self.cfg, self.expansion)
        moped_vectors = vectors(grad, self.likelihood.newcov)
        ycomp = moped_vectors.T @ self.likelihood.newdata
        return MOPEDstore(b_matrix=moped_vectors, ycomp=ycomp)

    def compression(self, cosmo: Cosmology) -> np.ndarray:
        """Given a test point in parameter space, we compress this to p numbers.

        Args:
            cosmo (Cosmology): the input test point.

        Returns:
            np.ndarray: the compressed theory at that point.
        """
        theory = self.likelihood.theory(cosmo)
        return self.store.b_matrix.T @ theory


def jla_moped_coefficients(
    compressor: JLAmoped, points: np.ndarray, cfg: ConfigDict
) -> np.ndarray:
    """Given the compressor, we will calculate the MOPED coefficients for each
    input parameter.

    Args:
        compressor (JLAmoped): the JLA compressor.
        points (np.ndarray): the input trraining points.
        cfg (ConfigDict): the main configuration file

    Returns:
        np.ndarray: the MOPED coefficients
    """
    points = np.atleast_2d(points)
    npoints, ndim = points.shape
    record_coeff = np.zeros((npoints, ndim))
    for i in range(npoints):
        parameters = get_jla_params(points[i], cfg)
        record_coeff[i] = compressor.compression(parameters)
    return record_coeff
