"""Aggregate data race detection dataset, tagged with the injection delay.

Like ``dataset.py``, but each record also carries a ``delay`` field. The delay
is the first element of the runtime's ``rtargs``: for each run, the module's
``parent`` is looked up in ``runtimes.json`` (found at the dataset root) and the
delay is read from ``metadata.rtargs[0]``.
"""

import json
import os

import numpy as np
from tqdm import tqdm


def _parse(p):
    p.add_argument("-p", "--path", help="Dataset path.")
    p.add_argument(
        "-v", "--violations", default="delay-sweep/violations.json",
        help="Aggregated violation indices.")
    p.add_argument("-o", "--out", help="Output base path.")
    return p


def dataset_iter(func, base: str) -> None:
    """Call a function for each (base_dir, device, run) in a dataset."""
    for device in tqdm(os.listdir(base), desc=base):
        if os.path.isdir(os.path.join(base, device)):
            for run in os.listdir(os.path.join(base, device)):
                func(base, device, run)


def _get_reentrant(benchmark, violations_path):
    with open(violations_path) as f:
        indices = np.array(json.load(f)[benchmark])
    return indices[:, 0] == indices[:, 1]


def _main(args):
    with open(args.violations) as f:
        violations = {
            benchmark: {tuple(k): i for i, k in enumerate(v)}
            for benchmark, v in json.load(f).items()}

    # Map each runtime uuid -> injection delay (first rtarg).
    with open(os.path.join(args.path, "runtimes.json")) as f:
        delays = {
            uuid: int(rt["metadata"]["rtargs"][0])
            for uuid, rt in json.load(f).items()}

    dataraces = {}

    def _load_data(base, device, run):
        with open(os.path.join(base, device, run)) as f:
            data = json.load(f)
        benchmark, density = data["module"]["name"].split('.')
        delay = delays[data["module"]["parent"]]

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
            "delay": np.array(delay, dtype=np.uint16),
            "bugs": np.packbits(bugs)})

    def _stack(d):
        return {k: np.array([x[k] for x in d]) for k in d[0]}

    dataset_iter(_load_data, args.path)
    stacked = {k: _stack(v) for k, v in dataraces.items()}

    os.makedirs(args.out, exist_ok=True)

    for k, v in stacked.items():
        np.savez(
            os.path.join(args.out, k + '.npz'),
            reentrant=_get_reentrant(k, args.violations),
            sites=np.array(list(violations[k].keys())),
            **v)
