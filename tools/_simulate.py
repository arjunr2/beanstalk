"""Debugging simulations."""

from multiprocessing.pool import Pool
import numpy as np
from jaxtyping import Bool, UInt


def simulate(
    t: UInt[np.ndarray, "samples"], bugs: Bool[np.ndarray, "samples bugs"],
    budget: int = 60 * 60 * 1000 * 1000
) -> int:
    """Run debugging simulation."""
    # Sample uniformly until reaching the compute budget
    order = np.arange(t.shape[0])
    np.random.shuffle(order)
    n = np.argmax(np.cumsum(t) > budget)

    # Make sure we haven't exhausted the dataset.
    if n == 0:
        return -1

    n_bugs = np.sum(np.any(bugs[order[:n]], axis=0) > 0)
    return n_bugs


def _simulate(args):
    return simulate(*args)


def simulate_pool(
    t: UInt[np.ndarray, "samples"], bugs: Bool[np.ndarray, "samples bugs"],
    budget: int = 60 * 60 * 1000 * 1000, processes: int = 32,
    samples: int = 1000
) -> list[int]:
    """Simulate (multithreaded)."""
    p = Pool(processes)
    return p.map(_simulate, ((t, bugs, budget) for _ in range(samples)))
