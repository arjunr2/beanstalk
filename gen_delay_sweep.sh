#!/bin/bash

mkdir -p delay-sweep
# Generate list of violations
python3 manage.py violations     -p delay-sweep-raw -o delay-sweep/violations.json;
# Generate data NPZ files from raw
python3 manage.py delay_dataset   -p delay-sweep-raw -o delay-sweep/baseline
