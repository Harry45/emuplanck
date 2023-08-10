from ml_collections.config_dict import ConfigDict


def get_config(experiment) -> ConfigDict:
    config = ConfigDict()
    config.logname = "planck-2018"
    config.experiment = experiment

    # cosmological parameters
    config.cosmo = cosmo = ConfigDict()
    cosmo.names = ["ombh2", "omch2", "thetastar", "tau", "As", "ns"]

    # emulator settings
    config.emu = emu = ConfigDict()
    emu.jitter = 1e-10
    emu.lr = 0.01
    emu.nrestart = 5
    emu.niter = 1000

    return config
