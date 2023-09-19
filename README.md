## Emulator for the Planck Likelihood

We build an emulator for the Planck likelihood. This is based on the work done by Prince et al ([paper](), [code](https://github.com/heatherprince/planck-lite-py)).


To run the code, we can do the following:

```
python main.py --config=config.py:planck-2018 --config.sampling.nsamples=5 --config.sampling.fname=test_1 --config.sampling.uniform_prior=True
```
