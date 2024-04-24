"""
Code: Main script for joint analysis.
Date: April 2024
Author: Arrykrishna
"""

# pylint: disable=bad-continuation
import os
import numpy as np
from absl import flags, app
from ml_collections.config_flags import config_flags
from multiprocessing import cpu_count
import warnings

# our script
from src.emulike.jointanalysis.sampling import sample_joint, sample_posterior
from utils.logger import get_logger

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("configjla", None, "Main configuration file for JLA.")
config_flags.DEFINE_config_file(
    "configplanck", None, "Main configuration file for Planck."
)
PATH = os.path.dirname(os.path.realpath(__file__))
warnings.filterwarnings("ignore")


def main(_):
    """
    Run the main sampling code and stores the samples.
    """
    cfg_jla = FLAGS.configjla
    cfg_planck = FLAGS.configplanck
    ncpu = cpu_count()

    logger = get_logger(cfg_planck, PATH)
    logger.info("Running main script")
    logger.info(f"We have {ncpu} CPUs")
    sampler = sample_posterior(cfg_jla, cfg_planck)


if __name__ == "__main__":
    app.run(main)
