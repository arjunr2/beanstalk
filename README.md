# Beanstalk Data and Processing Artifacts [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14933491.svg)](https://doi.org/10.5281/zenodo.14933491)

This repo contains all the data and tools needed to process data for Beanstalk. The dataset can be found on [Zenodo](https://doi.org/10.5281/zenodo.14927586) here. See the [artifact overview](ARTIFACT-OVERVIEW.md) for usage instructions and [dataset explainer](DATASET-EXPLAINER.md) for details on dataset structure if necessary.

## Environment Setup
Python with JAX, `matplotlib`, and `tqdm` packages are needed.
GPU backend for JAX is not required, but highly recommended to run processing scripts quickly. If you have these already, you can skip the rest of this section


The environment packaged as Docker build for consistency:
```bash
# Defaults to GPU enabled systems.
# Use the 'Dockerfile.cpu' build file if using CPU fallback
docker build -t beanstalk .
```

If using the Docker environment, run this in the root of this repo directory before proceeding with following steps:

```bash
# If using CPU fallback, remove '--gpus all'
docker run -it --gpus all --rm -v "`pwd`:/home/evaluator" beanstalk
```



## Generating Figures (From Pre-Processed Data)

Extract `{data,summary,simulations}.zip` packaged in Zenodo/Github releases to the root directory of this repo.

To generate figures, run `./gen_figures.sh` (ignore any runtime warnings). The output should mirror `figures.zip` . 


## Reproducing Results from Raw Data

The contained directories  `data`, `summary`, and `simulations` above can be reproduced from the raw cluster data, following these steps:

1. **Extract Raw Data** from `data-raw.zip` to the root directory of the repo.
2. **Data**: Run `./gen_data.sh` (approx. 2 min to run).
3. **Summary**: Run `./summarize.sh` (approx. 2 min to run on GPU).
4. **Simulations**: Run `./run_simulations.sh` (approx. 20 min to run 10000 replicates on GPU). If necessary, replicates can be configured with first argument to the script.
5. **Figures**: Run `./gen_figures.sh` (as in the [previous section](#generating-figures-from-pre-processed-data)).

> **NOTES**: 
> * GPU support for JAX is recommended; CPU backends can be alternatively be used, but may take significantly longer to execute.
> * The `manage.py` script manages all data scripts (see `-h` option for more information on what parameters can be configured for experiments).
