"""Get unique violations."""

import json
import os

from tqdm import tqdm


def _parse(p):
    p.add_argument(
        "-p", "--path", nargs='+', default=[], help="Dataset paths.")
    p.add_argument(
        "-o", "--out", default="data/violations.json", help="Output path.")
    return p


def dataset_iter(func, base: str) -> None:
    """Call a function for each (base_dir, device, run) in a dataset."""
    for device in tqdm(os.listdir(base), desc=base):
        if os.path.isdir(os.path.join(base, device)):
            for run in os.listdir(os.path.join(base, device)):
                func(base, device, run)


def _main(args):

    violations = {}

    def _get_unique_violations(base, device, run):
        with open(os.path.join(base, device, run)) as f:
            data = json.load(f)
        benchmark, _ = data["module"]["name"].split('.')

        if benchmark not in violations:
            violations[benchmark] = set()

        for v in data['violations']:
            # violation convention: lower-index bug is always first
            key = (min(v['i1'], v['i2']), max(v['i1'], v['i2']))
            if key not in violations[benchmark]:
                violations[benchmark].add(key)

    for p in args.path:
        dataset_iter(_get_unique_violations, p)

    with open(args.out, 'w') as f:
        json.dump({k: list(v) for k, v in violations.items()}, f)
