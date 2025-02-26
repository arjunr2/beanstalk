#!/bin/bash

mkdir -p simulations
set -x
# Compute budget simulations
python3 manage.py simulate -p data/baseline -o simulations/baseline.npz
python3 manage.py simulate -p data/beanstalk -o simulations/beanstalk.npz
python3 manage.py simulate -p data/beanstalk -a density -o simulations/abl_density.npz
python3 manage.py simulate -p data/beanstalk -a device -o simulations/abl_device.npz
# Max density simulations
python3 manage.py simulate2 -p data/beanstalk -o simulations/density.npz
set +x
