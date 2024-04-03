import os
import numpy as np
import pandas as pd
from numpyro.diagnostics import summary
from ml_collections.config_dict import ConfigDict
from utils.helpers import pickle_load, pickle_save, emcee_chains
from utils.MCEvidence import MCEvidence


def emcee_stats(
    cfg: ConfigDict, fname1: str, fname2: str, discard: int = 1000, thin: int = 2
) -> pd.DataFrame:

    mcmc_1 = pickle_load(cfg.path.parent, fname1)
    mcmc_2 = pickle_load(cfg.path.parent, fname2)

    nevals = mcmc_1.flatchain.shape[0] + mcmc_2.flatchain.shape[0]

    samples_1 = mcmc_1.get_chain(discard=discard, thin=thin, flat=True)
    samples_2 = mcmc_2.get_chain(discard=discard, thin=thin, flat=True)

    print(f"Number of samples for chain 1: {mcmc_1.flatchain.shape[0]}")
    print(f"Number of samples for chain 2: {mcmc_2.flatchain.shape[0]}")
    print(f"Number of samples for chain 1 (after processing): {samples_1.shape[0]}")
    print(f"Number of samples for chain 2 (after processing): {samples_2.shape[0]}")

    record = []
    for i, key in enumerate(cfg.sampling.names):
        testsamples = np.vstack(([samples_1[:, i], samples_2[:, i]]))
        summary_stats = summary(testsamples)
        summary_stats[key] = summary_stats.pop("Param:0")
        record.append(summary_stats)

    record_df = []
    for i in range(len(record)):
        record_df.append(
            pd.DataFrame(record[i]).round(4).loc[["r_hat", "n_eff", "mean", "std"]]
        )

    record_df = pd.concat(record_df, axis=1).T
    record_df["scaled_n_eff"] = record_df["n_eff"] / nevals
    return record_df


def calculate_evidence(
    cfg: ConfigDict,
    fname1: str,
    fname2: str,
    thin: int = 2,
    discard: int = 1000,
    MCEthin=0.5,
) -> float:
    """Calculates the Evidence from MCMC samples using MCEvidence.

    Args:
        cfg (ConfigDict): the main configuration file.
        fname1 (str): name of the first MCMC file.
        fname2 (str): name of the second MCMC file.
        thin (int): the thinning factor to use with EMCEE
        discard (int): the number of samples to discard per walker in EMCEE
        MCEthin (float, optional): If 0<thinlen<1, MCMC weights are adjusted based on
        Poisson sampling. Defaults to 0.5.

    Returns:
        float: the evidence value
    """
    path, file = os.path.split(fname1)
    chainfile = os.path.join(path, "MCE_" + file[:-1] + "*")

    mce_1 = emcee_chains(cfg, fname1, thin, discard)
    mce_2 = emcee_chains(cfg, fname2, thin, discard)

    evi_sim = MCEvidence(
        chainfile,
        split=True,
        kmax=0,
        verbose=0,
        priorvolume=1.0,
        thinlen=MCEthin,
        burnlen=0,
        debug=False,
    )
    evidence = evi_sim.evidence()[0]
    return evidence
