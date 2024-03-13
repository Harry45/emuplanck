## Emulator for the Planck Likelihood

We build an emulator for the Planck likelihood. This is based on the work done by Prince et al ([paper](), [code](https://github.com/heatherprince/planck-lite-py)).


To run the code, we can do the following:

### Planck

```
python3 -m src.emulike.planck.main --config=src/emulike/planck/config.py:planck-2018 --config.sampling.nsamples=1 --config.sampling.fname=test_1 --config.sampling.uniform_prior=True
```

### JLA
```
time python3 -m src.emulike.jla.main --config=src/emulike/jla/config.py:jla --config.sampling.nsamples=5000 --config.sampling.fname=emulator_1
```