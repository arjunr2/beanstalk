#!/bin/bash

mkdir -p data
# Generate list of violations
python3 manage.py violations -p data-raw/beanstalk
# Generate data NPZ files from raw
python3 manage.py dataset -p data-raw/beanstalk -o data/beanstalk
python3 manage.py dataset -p data-raw/beanstalk -o data/beanstalk

