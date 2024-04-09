import os
import logging
import numpy as np
from datetime import datetime
import emcee
from ml_collections.config_dict import ConfigDict

from src.emulike.jla.distribution import jla_loglike_sampler
from src.emulike.planck.distribution import (
    planck_loglike_sampler,
    planck_logprior_normal,
)

from src.emulike.jla.sampling import get_jla_priors_emulator
from src.emulike.planck.sampling import get_priors_emulator

from experiments.jla.jlalite import JLALitePy
from experiments.planck.plite import PlanckLitePy
from utils.helpers import get_jla_planck_fname, pickle_save

LOGGER = logging.getLogger(__name__)
PATH = os.path.dirname(os.path.realpath(__file__))


def sample_joint(
    parameters,
    cfg_jla,
    like_jla,
    priors_jla,
    emu_jla,
    cfg_planck,
    like_planck,
    priors_planck,
    emu_planck,
) -> float:

    loglike_planck = planck_loglike_sampler(
        parameters, like_planck, cfg_planck, priors_planck, emu_planck
    )
    loglike_jla = jla_loglike_sampler(
        parameters[0:3], like_jla, cfg_jla, priors_jla, emu_jla
    )

    logprior = planck_logprior_normal(parameters, priors_planck)

    return loglike_planck + loglike_jla + logprior


def sample_posterior(cfg_jla, cfg_planck):
    priors_jla, emu_jla = get_jla_priors_emulator(cfg_jla)
    like_jla = JLALitePy(cfg_jla)

    priors_planck, emu_planck = get_priors_emulator(cfg_planck)
    like_planck = PlanckLitePy(
        data_directory=cfg_planck.path.data,
        year=cfg_planck.planck.year,
        spectra=cfg_planck.planck.spectra,
        use_low_ell_bins=cfg_planck.planck.use_low_ell_bins,
    )

    if cfg_planck.sampling.run_sampler and cfg_jla.sampling.run_sampler:
        pos = cfg_planck.sampling.mean + 1e-4 * np.random.normal(
            size=(2 * cfg_planck.ndim, cfg_planck.ndim)
        )
        nwalkers = pos.shape[0]
        start_time = datetime.now()
        sampler = emcee.EnsembleSampler(
            nwalkers,
            cfg_planck.ndim,
            sample_joint,
            args=(
                cfg_jla,
                like_jla,
                priors_jla,
                emu_jla,
                cfg_planck,
                like_planck,
                priors_planck,
                emu_planck,
            ),
        )
        sampler.run_mcmc(pos, cfg_planck.sampling.nsamples, progress=True)

        time_elapsed = datetime.now() - start_time
        LOGGER.info(f"Time: sample the posterior : {time_elapsed}")

        # save the sampler
        fname = get_jla_planck_fname(cfg_jla, cfg_planck)
        path = os.path.join(PATH, "samples")
        pickle_save(sampler, path, fname)
        LOGGER.info(f"Total number of samples: {sampler.flatchain.shape}")

    return sampler
