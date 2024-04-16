#!/bin/bash
source /mnt/zfsusers/phys2286/anaconda3/etc/profile.d/conda.sh
source activate emuplanck
echo $PWD
which python
echo $(for i in $(seq 1 50); do printf "-"; done)
python3 -m src.emulike.jointanalysis.main --configjla=configs/JLAconfig.py:jla --configplanck=configs/Planckconfig.py:planck-2018 \
    --configplanck.sampling.nsamples=10000 --configplanck.sampling.fname=experiment_2 --configplanck.logname=jointanalysis
# python3 -m src.emulike.planck.main --config=configs/Planckconfig.py:planck-2018 \
#     --config.sampling.nsamples=10000 --config.sampling.fname=experiment_2
# python3 -m src.emulike.jla.main --config=configs/JLAconfig.py:jla \
#     --config.sampling.nsamples=10000 --config.sampling.fname=experiment_2
# run script with the following
# addqueue -m 16 -n 1x8 -s ./runme.sh
