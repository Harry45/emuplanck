"""
Code: Main script for sampling the Planck Lite likelihood.
Date: August 2023
Author: Arrykrishna
"""

# pylint: disable=bad-continuation
import os
from absl import flags, app
from ml_collections.config_flags import config_flags
from multiprocessing import cpu_count
import warnings

# our scripts and functions
from src.emulike.planck.sampling import sample_posterior
from utils.logger import get_logger

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", None, "Main configuration file.")
PATH = os.path.dirname(os.path.realpath(__file__))
warnings.filterwarnings("ignore")


def main(_):
    """
    Run the main sampling code and stores the samples.
    """
    cfg = FLAGS.config
    ncpu = cpu_count()
    logger = get_logger(FLAGS.config, PATH)
    logger.info("Running main script")
    logger.info(f"We have {ncpu} CPUs")

    # run the sampler
    sample_posterior(cfg)


if __name__ == "__main__":
    app.run(main)
