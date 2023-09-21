#!/bin/bash
source /mnt/zfsusers/phys2286/anaconda3/etc/profile.d/conda.sh
conda activate jaxcosmo
which python
echo $(for i in $(seq 1 50); do printf "-"; done)
python main.py --config=config.py:planck-2018 --config.sampling.nsamples=10000 --config.sampling.uniform_prior=True --config.sampling.fname=planck_priors_2

# run script with the following
# addqueue -m 16 -n 1x8 -s ./runme.sh