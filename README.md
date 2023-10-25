# Beanstalk


## Dataset

Our dataset is organized as follows:

```sh
data/
    violations.json     # list of unique violations discovered
    beanstalk/          # data collected for our method
        indirect.npz    # different file for each benchmark
        ...
    baseline/           # data used for our 100% instrumented baseline
        ...
```

Dataset files have the following arrays:
- `t: uint32[N]`: Execution time (in microseconds) of each run
- `device: uint8[N]`: Device that each run was executed on
- `density: uint8[N]`: Instrumentation density (%) for each run
- `bugs: uint8[N, ceil(b/8)] --> bool[N, b]`: Packed bit array indicating whether each bug was discovered on each run. Unpack using `np.unpackbits(arr, axis=1)`.
