## Emulator for the Planck Likelihood

We build an emulator for the Planck likelihood. This is based on the work done by Prince et al ([paper](https://arxiv.org/abs/1909.05869), [code](https://github.com/heatherprince/planck-lite-py)).


All boolean field are already set in the config file and to run the code, we can do the following:

### Planck (log-likelihood)

```
python3 -m src.emulike.planck.main --config=configs/Planckconfig.py:planck-2018 --config.sampling.nsamples=1 --config.sampling.fname=experiment_1
```

### JLA (log-likelihood)

```
python3 -m src.emulike.jla.main --config=configs/JLAconfig.py:jla --config.sampling.nsamples=1 --config.sampling.fname=experiment_1
```

### Planck and JLA (log-likelihood)

```
python3 -m src.emulike.jointanalysis.main --configjla=configs/JLAconfig.py:jla --configplanck=configs/Planckconfig.py:planck-2018 --configplanck.sampling.nsamples=10 --configplanck.sampling.fname=experiment_1 --configplanck.logname=jointanalysis

```

### JLA (MOPED Coefficients)

```
python3 -m src.moped.jla.main --config=configs/JLAconfig.py:jla-moped --config.sampling.nsamples=1 --config.sampling.fname=experiment_1
```

### Planck (MOPED Coefficients)

```
python3 -m src.moped.planck.main --config=configs/Planckconfig.py:planck-moped --config.sampling.nsamples=1 --config.sampling.fname=experiment_1
```

### To Do
- PCA - Planck (Power Spectra)
- PCA - JLA (Distance Modulus)
- Joint Analysis (with any combination - likelihood, MOPED or PCA)