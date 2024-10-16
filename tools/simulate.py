"""Run debugging budget simulations."""

import os
from tqdm import tqdm
import numpy as np

from ._simulate import simulate_pool


def _parse(p):
    p.add_argument("-p", "--path", help="Dataset path.")
    p.add_argument("-o", "--out", help="Output file.")
    p.add_argument(
        "--budget", type=int, nargs='+', help="Simulation budgets (minutes).",
        default=[1, 2, 5, 10, 15, 30, 60])
    p.add_argument(
        "-r", "--replicates", help="Number of simulations to run.",
        default=10000, type=int)
    p.add_argument("-a", "--ablation", default=None, help="Ablation to run.")


def _main(args):

    minutes = 60 * 1000 * 1000
    benchmarks = os.listdir(args.path)

    def _run(path):
        npz = np.load(path)
        t = npz['t']
        bugs = np.unpackbits(
            npz['bugs'], axis=1, count=npz['reentrant'].shape[0])

        if args.ablation == "density":
            mask = (
                (npz["device"] == 33) | (npz["device"] == 34)
                | (npz["device"] == 35))
            t = t[mask]
            bugs = bugs[mask]
        elif args.ablation == "device":
            mask = (npz["density"] == 100)
            t = t[mask]
            bugs = bugs[mask]

        return np.array([
            simulate_pool(t, bugs, budget=x * minutes, samples=args.replicates)
            for x in tqdm(args.budget, desc=path)])

    results = {b: _run(os.path.join(args.path, b)) for b in benchmarks}
    np.savez(args.out, **results)
