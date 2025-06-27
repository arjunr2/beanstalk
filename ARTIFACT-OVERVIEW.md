# Artifact Overview: Unveiling Heisenbugs with Diversified Execution

Authors:
* [Arjun Ramesh](arjunr2@andrew.cmu.edu)
* [Tianshu Huang](tianshu2@andrew.cmu.edu)
* [Jaspreet Riar](riarjaspreet77@gmail.com)
* [Ben L. Titzer](btitzer@andrew.cmu.edu)
* [Anthony Rowe](agr@andrew.cmu.edu)


## Introduction

This document is intended to facilitate the reuse and reproduction of experiments for the paper [**Unveiling Heisenbugs with Diversified Execution**](https://dl.acm.org/doi/pdf/10.1145/3720428). 
Users should be able to reuse the data and produce the results demonstrated in the paper.

### Paper Claims Backed By Artifact

The table below highlights *all* claims made by the paper and its supporting experiments. For any `field` in **Reference in Artifact**, the processing script is located in `plot/{field}.py` and the corresponding generated figure in `figures/{field}.pdf` of *Scripts* repo after running experiments

| #  | Claim                                                                                                                                                                                                   | Section in Paper                                                              | Figure in Paper | Reference in Artifact                                                         |
|----|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------|-----------------|-------------------------------------------------------------------------------|
| C1 | Heisenbugs can appear and disappear, even with minor changes in instrumentation                                                                                                                         | Sec. 1                                                                        | Fig. 1          | `hero`                                                                        |
| C2 | Heisenbugs are difficult to pinpoint by inspection, and can be finely interleaved with ordinary bugs                                                                                                    | Sec. 1                                                                        | Fig. 2          | `code_examples` <br>(code snippet located in `code_examples/code_examples.c`) |
| C3 | Many detected bugs exhibit heisenbug properties with highly susceptible observability                                                                                                                   | Sec. 5.3 (Prevalence of Heisenbugs)                                           | Fig. 9          | `hfactor_marginals`                                                           |
| C4 | While bugs show a negative correlation between Heisen Factor and their observability (common intuition that rarer bugs tend to heisenbugs), *some heisenbugs can have surprisingly high observability*. | Sec. 5.3 (Heisen Factor and Observability)                                    | Fig. 10         | `hfactor_observability`                                                       |
| C5 | Increased bug observability can compensate for reduced instrumentation to produce higher bug detectability that is exploitable by debuggers like Beanstalk                                              | Sec. 5.3 (Increased observability can compensate for reduced instrumentation) | Fig. 11         | `example`                                                                     |
| C6 | Bugs show a wide range of detectability profiles patterns that get increasingly unexplainable with increasing Heisen Factors                                                                            | Sec. 5.3 (Increased observability can compensate for reduced instrumentation) | Fig. 12         | `poster`                                                                      |
| C7 | Many bugs (nearly 30%) are undetectable without both instrumentation and platform diversity leveraged by Beanstalk                                                                                      | Sec. 5.4 (Detection Probability)                                              | Fig. 13         | `cdf`                                                                         |
| C8 | Beanstalk finds more bugs on average than homogeneous baselines on debugging campaigns, with an especially better 5% worse-case and 95% best case                                                       | Sec. 5.4 (Compute Budget)                                                     | Fig. 14         | `simulation_budget`                                                           |
| C9 | Beanstalk can find more bugs than the homogeneous baseline even under low instrumentation densities. This allows it to operate at exceedingly low overheads at a large scale   in                       | Sec. 5.4 (Instrumentation Density)                                            | Fig. 15 + 16    | `simulation_density` + `overhead`                                             |

#### What is omitted?
Notably, the *raw dataset* was captured on the hardware cluster shown in Fig. 8, and is hence not easily reproducible. However, the software we used to capture this data is open-sourced at https://github.com/SilverLineFramework/runtime-manager/tree/datarace, and can be used by anyone with a similar setup in the future to generate raw data.


## Hardware Dependencies

While no special hardware is absolutely necessary, a Nvidia GPU system is **highly recommended** to accelerate data processing. CPU fallback can take up to 10x longer to run.

The total artifact will require the following:
* $\approx$ 20 GB of disk
* $\geq$ 8GB of RAM is preferable
* $\geq$ 4 cores is preferable


## Getting Started Guide

> **NOTE**: The instructions are fully self-contained and described in detail within the READMEs of the artifact repositories below. The steps in rest of the document hence mostly reiterates the content available there.

You need two software repositories:
* *Dataset*: https://doi.org/10.5281/zenodo.14927586
* *Scripts*: https://github.com/arjunr2/beanstalk (release v2.0.4). This is also mirrored at https://doi.org/10.5281/zenodo.14933491

To start, do the following:

1. Download/unzip the *Scripts* repository. 
2. After completed, download and unzip the contents of *Dataset* repository in the root directory of the *Scripts* repository. The dataset is quite large, and may take a while to download/unzip -- you may proceed with the next step of installing software dependencies during this time.

#### Software Dependencies

You need Python 3, with support for `jax` (preferably GPU backend), `matplotlib`, and `tqdm` to run experiments. The easiest way to get this is with [Docker](https://docs.docker.com/engine/install/).
> To additionally enable GPU support, you will need the [Nvidia Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

> Docker can be up to 2x slower than native. If you have a native, functional Python setup, feel free to use that, but we cannot guarantee a smooth experience.


The *Scripts* repo contains Dockerfiles for convenience. To build the Docker image, run:
```bash
# Defaults to GPU enabled systems.
# Use the 'Dockerfile.cpu' build file if using CPU fallback
docker build -t beanstalk .
```

To open a shell with the environment, run (within the root of the *Scripts* repo):
```bash
# If using CPU fallback, remove '--gpus all'
docker run -it --gpus all --rm -v "`pwd`:/home/evaluator" beanstalk
```

If this has taken you less than 30 minutes, you can proceed to run artifacts.

## Step by Step Instructions

The `data-raw.zip` consists of the cluster data that can be used to generate all figures/experiments in the paper.

### Generating Figures (From Pre-Processed Data)

The `data.zip`, `summary.zip`, `simulations.zip` in *Dataset* contain the exact data used by us for the paper. 

To re-generate figures:
1. Extract the above zips in the root directory of *Scripts*
2. Run `./gen_figures.sh` (ignore any runtime warnings). The output should replicate `figures.zip`.


### Reproducing Results from Raw Data

The  `data`, `summary`, and `simulations` above can be reproduced from the raw cluster data, following these steps:

1. **Extract Raw Data** from `data-raw.zip` to the root directory of *Scripts*
2. **Data**: Run `./gen_data.sh` (approx. 2 min to run).
3. **Summary**: Run `./summarize.sh` (approx. 2 min to run on GPU).
4. **Simulations**: Run `./run_simulations.sh` (approx. 20 min to run 10000 replicates on GPU). If necessary, replicates can be configured with first argument to the script.
5. **Figures**: Run `./gen_figures.sh` (as in the [previous section](#generating-figures-from-pre-processed-data)).

The relevant results to verify each individual claim is listed at the start of this document ([Paper Claims Backed By Artifact](#paper-claims-backed-by-artifact))

## Reusability Guide

All of the software here should be reusable and self-documented
*  README markdowns in the repositories describe detailed usage beyond those above.
* The `manage.py` script manages all data scripts (see -h option for more information on what parameters can be configured for experiments).
* There should be no limits to reusing this artifact.
* New inputs or test cases can be fed in by adhering to the data formats used by the scripts. 
