# Beanstalk Data and Processing Artifacts [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14933491.svg)](https://doi.org/10.5281/zenodo.14933491)

This repo contains all the data and tools needed to process data for Beanstalk

## Dataset Structure

All assets are included as `*.zip` files in releases. Our dataset is organized as follows:

```sh
data-raw/
    beanstalk/          # raw data from cluster collected for our method
    baseline/           # raw data used for our 100% instrumented baseline

data/
    violations.json     # list of unique violations discovered
    beanstalk/          # data collected for our method
        indirect.npz    # different file for each benchmark
        ...
    baseline/           # data used for our 100% instrumented baseline
        ...
```

Each benchmark file in `data` has the following arrays:
- `t: uint32[N]`: Execution time (in microseconds) of each run
- `device: uint8[N]`: Device that each run was executed on
- `density: uint8[N]`: Instrumentation density (%) for each run
- `bugs: uint8[N, ceil(b/8)] --> bool[N, b]`: Packed bit array indicating whether each bug was discovered on each run. Unpack using `np.unpackbits(arr, axis=1)`.
- `sites: uint32[b, 2]`: Pair of code indices (in Wasm module) responsible for the bug


### Processed Dataset Structure

The evaluation/processing scripts generate the following processed data: 
```sh
summary/                # statistical summarized 'data' directory
    indirect.npz        # different file for each benchmark
    ...

simulations/
    # Compute budget simulations
    abl_density.npz     # Density ablation
    abl_device.npz      # Device ablation
    baseline.npz        # Homogeneous baseline ablation
    beanstalk.npz       # Beanstalk ablation (unconstrained instrumentation density)
    # Maximum instrumentation density simulation
    density.npz         # Beanstalk ablation (constrained instrumentation density)

figures/                # PDF figures generated for the paper
```


## Generating Figures from Packaged Data

Extract  `{data,summary,simulations}.zip` packaged in the release to the root directory of this repo.

To generate figures, run `./gen_figures.sh` (ignore any runtime warnings). The output should mirror `figures.zip` . 


## Reproducing Results from Raw Data

The three zip files for data, summary, and simulations can be reproduced from the raw cluster data, following these steps:

1. **Extract Raw Data** from `data-raw.zip` to the root directory of the repo.
2. **Data**: Run `./gen_data.sh` (approx. 2 min to run).
3. **Summary**: Run `./summarize.sh` (approx. 2 min to run on GPU).
4. **Simulations**: Run `./run_simulations.sh` (approx. 20 min to run 10000 replicates on GPU). If necessary, replicates can be configured with first argument to the script.
5. Generate figures as in [previous section](#generating-figures-from-pre-packaged-data).

> **NOTES**: 
> * GPU support for JAX is recommended; CPU backends can be alternatively be used, but may take significantly longer to execute.
> * The `manage.py` script manages all data scripting (see `-h` option for more information) .
