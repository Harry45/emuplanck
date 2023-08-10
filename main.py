"""
Code: Main script for sampling the Planck Lite likelihood.
Date: August 2023
Author: Arrykrishna
"""
# pylint: disable=bad-continuation
from absl import flags, app
from ml_collections.config_flags import config_flags


# our scripts and functions
from src.sampling import sample_posterior

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", None, "Main configuration file.")


def main(args):
    """
    Run the main sampling code and stores the samples.
    """
    cfg = FLAGS.config

    # run the sampler
    sample_posterior(cfg)


if __name__ == "__main__":
    app.run(main)
