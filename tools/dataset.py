"""Aggregate data race detection dataset."""

import json
import os

import numpy as np
from tqdm import tqdm


def _parse(p):
    p.add_argument("-p", "--path", help="Dataset path.")
    p.add_argument(
        "-v", "--violations", default="data/violations.json",
        help="Aggregated violation indices.")
    p.add_argument("-o", "--out", help="Output base path.")
    return p


def dataset_iter(func, base: str) -> None:
    """Call a function for each (base_dir, device, run) in a dataset."""
    for device in tqdm(os.listdir(base), desc=base):
        if os.path.isdir(os.path.join(base, device)):
            for run in os.listdir(os.path.join(base, device)):
                func(base, device, run)


def _get_reentrant(benchmark):
    with open("data/violations.json") as f:
        indices = np.array(json.load(f)[benchmark])
    return indices[:, 0] == indices[:, 1]


def _main(args):
    with open("data/violations.json") as f:
        violations = {
            benchmark: {tuple(k): i for i, k in enumerate(v)}
            for benchmark, v in json.load(f).items()}

    dataraces = {}

    def _load_data(base, device, run):
        with open(os.path.join(base, device, run)) as f:
            data = json.load(f)
        benchmark, density = data["module"]["name"].split('.')

        bugs = np.zeros(len(violations[benchmark]), dtype=bool)
        for v in data['violations']:
            # violation convention: lower-index bug is always first
            key = (min(v['i1'], v['i2']), max(v['i1'], v['i2']))
            bugs[violations[benchmark][key]] = True

        if benchmark not in dataraces:
            dataraces[benchmark] = []
        dataraces[benchmark].append({
            "t": np.array(data['cpu_time'], dtype=np.uint32),
            "device": np.array(device.split('-')[1], dtype=np.uint8),
            "density": np.array(density, dtype=np.uint8),
            "bugs": np.packbits(bugs)})

    def _stack(d):
        return {k: np.array([x[k] for x in d]) for k in d[0]}

    dataset_iter(_load_data, args.path)
    stacked = {k: _stack(v) for k, v in dataraces.items()}

    os.makedirs(args.out, exist_ok=True)

    for k, v in stacked.items():
        np.savez(
            os.path.join(args.out, k + '.npz'),
            reentrant=_get_reentrant(k), sites=np.array(list(violations[k].keys())),
            **v)
