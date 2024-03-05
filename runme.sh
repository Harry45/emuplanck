#!/bin/bash
module load cuda/12.3
source /mnt/zfsusers/phys2286/anaconda3/etc/profile.d/conda.sh
source activate jaxcosmo
echo $PWD
which python
echo $(for i in $(seq 1 50); do printf "-"; done)
python3 -m src.emulike.planck.main --config=src/emulike/planck/config.py:planck-2018 \
    --config.sampling.nsamples=1 --config.sampling.fname=test_1 --config.sampling.uniform_prior=True

# run script with the following
# addqueue -m 16 -n 1x8 -s ./runme.sh
