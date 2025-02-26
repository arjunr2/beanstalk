#!/bin/bash

REPLICATES=${1:-10000}

mkdir -p simulations

set -x
# Compute budget simulations
python3 manage.py simulate -p data/baseline -o simulations/baseline.npz -r $REPLICATES
python3 manage.py simulate -p data/beanstalk -o simulations/beanstalk.npz -r $REPLICATES
python3 manage.py simulate -p data/beanstalk -a density -o simulations/abl_density.npz -r $REPLICATES
python3 manage.py simulate -p data/beanstalk -a device -o simulations/abl_device.npz -r $REPLICATES
# Max density simulations
python3 manage.py simulate2 -p data/beanstalk -o simulations/density.npz -r $REPLICATES
set +x
