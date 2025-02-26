# Beanstalk Data and Processing Artifacts

This repo contains all the data and tools needed to process data for Beanstalk

## Dataset

Our dataset is organized as follows:

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

summary/                # statistical summarized 'data' directory
    indirect.npz        # different file for each benchmark
    ...

figures/                # PDF figures generated for the paper

simulations/
    # Compute budget simulations
    abl_density.npz     # Density ablation
    abl_device.npz      # Device ablation
    baseline.npz        # Homogeneous baseline ablation
    beanstalk.npz       # Beanstalk ablation (unconstrained instrumentation density)
    # Maximum instrumentation density simulation
    density.npz         # Beanstalk ablation (constrained instrumentation density)
```

Each benchmark file in `data` has the following arrays:
- `t: uint32[N]`: Execution time (in microseconds) of each run
- `device: uint8[N]`: Device that each run was executed on
- `density: uint8[N]`: Instrumentation density (%) for each run
- `bugs: uint8[N, ceil(b/8)] --> bool[N, b]`: Packed bit array indicating whether each bug was discovered on each run. Unpack using `np.unpackbits(arr, axis=1)`.
- `sites: uint32[b, 2]`: Pair of code indices (in Wasm module) responsible for the bug

 