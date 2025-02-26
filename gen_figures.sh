#!/bin/bash

mkdir -p figures

set -x
# Figure 1
python3 plot/hero.py

# Figure 2
python3 plot/code_examples.py

# Figure 9
python3 plot/hfactor_marginals.py

# Figure 10
python3 plot/hfactor_observability.py

# Figure 11
python3 plot/example.py

# Figure 12
python3 plot/poster.py

# Figure 13
python3 plot/cdf.py

# Figure 14
python3 plot/simulation_budget.py

# Figure 15
python3 plot/simulation_density.py

# Figure 16
python3 plot/overhead.py

set +x
