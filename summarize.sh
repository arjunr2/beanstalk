#!/bin/bash

mkdir -p summary
for x in data/beanstalk/*.npz; do 
  echo "Summarizing: $x"
  python3 manage.py summarize -p $x -o summary/$(basename $x); 
done
