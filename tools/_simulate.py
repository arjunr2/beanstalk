"""Debugging simulations."""

from multiprocessing.pool import Pool
import numpy as np
from jaxtyping import Bool, UInt, Int8


def simulate(
    t: UInt[np.ndarray, "samples"], bugs: Bool[np.ndarray, "samples bugs"],
    budget: int = 60 * 60 * 1000 * 1000
) -> Int8[np.ndarray, "bugs"]:
    """Run debugging simulation. Returns [-1 ...] as an error value."""
    # Sample uniformly until reaching the compute budget
    order = np.arange(t.shape[0])
    np.random.shuffle(order)
    n = np.argmax(np.cumsum(t) > budget)

    # Make sure we haven't exhausted the dataset.
    if n == 0:
        return np.full(bugs.shape[1], -1, dtype=np.int8)
    else:
        return np.any(bugs[order[:n]], axis=0).astype(np.int8)


def _simulate(args):
    return simulate(*args)


def simulate_pool(
    t: UInt[np.ndarray, "samples"], bugs: Bool[np.ndarray, "samples bugs"],
    budget: int = 60 * 60 * 1000 * 1000, processes: int = 32,
    samples: int = 1000
) -> list[Int8[np.ndarray, "bugs"]]:
    """Simulate (multithreaded)."""
    p = Pool(processes)
    return p.map(_simulate, ((t, bugs, budget) for _ in range(samples)))
