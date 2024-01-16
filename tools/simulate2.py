"""Run debugging max_density simulations."""

import os
from tqdm import tqdm
import numpy as np

from ._simulate import simulate_pool


def _parse(p):
    p.add_argument("-p", "--path", help="Dataset path.")
    p.add_argument("-o", "--out", help="Output file.")
    p.add_argument(
        "--density", type=int, nargs='+', help="Maximum density (percent).",
        default=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    p.add_argument(
        "-r", "--replicates", help="Number of simulations to run.",
        default=10000, type=int)


def _main(args):

    minutes = 60 * 1000 * 1000
    budget = 30 * minutes
    benchmarks = os.listdir(args.path)

    def _run_filter(density, t, bugs, max_density):
        mask = (density <= max_density)
        return simulate_pool(
            t[mask], bugs[mask], budget=budget, samples=args.replicates)

    def _run(path):
        npz = np.load(path)
        bugs = np.unpackbits(
            npz['bugs'], axis=1, count=npz['reentrant'].shape[0])
        return np.array([
            _run_filter(npz['density'], npz['t'], bugs, x)
            for x in tqdm(args.density, desc=path)])

    results = {b: _run(os.path.join(args.path, b)) for b in benchmarks}
    np.savez(args.out, **results)
