"""Calculate summary statistics."""

import numpy as np
from jax import random, vmap
from jax import numpy as jnp
import json

from jaxtyping import UInt, Bool, Array, Float


def _parse(p):
    p.add_argument("-p", "--path", help="Path to benchmark in dataset.")
    p.add_argument("-o", "--out", help="Output file.")
    p.add_argument("-s", "--seed", help="Random seed.", type=int, default=42)
    p.add_argument(
        "--quantile", type=float, nargs='+',
        default=[0.005, 0.01, 0.025, 0.05, 0.5, 0.95, 0.975, 0.99, 0.995],
        help="Confidence interval bounds.")
    p.add_argument(
        "--samples", type=int, default=10000,
        help="Number of samples for monte carlo simulation.")
    return p


def _get_summary_matrix(path) -> tuple[
    UInt[np.ndarray, "Nv Nd Nb"],
    UInt[np.ndarray, "Nv Nd"],
    UInt[np.ndarray, "Nv"],
    UInt[np.ndarray, "Nd"],
    Bool[np.ndarray, "Nb"]
]:
    npz = np.load(path)
    devices: UInt[np.ndarray, "Nv"] = np.sort(np.unique(npz["device"]))
    densities: UInt[np.ndarray, "Nd"] = np.sort(np.unique(npz["density"]))
    reentrant: Bool[np.ndarray, "Nb"] = npz["reentrant"]
    bugs: Bool[np.ndarray, "N Nb"] = np.unpackbits(
        npz["bugs"], axis=1, count=reentrant.shape[0])

    shape = (len(devices), len(densities))
    summary: UInt[np.ndarray, "Nv Nd Nb"] = np.zeros(
        (*shape, bugs.shape[1]), dtype=np.uint32)
    runs: UInt[np.ndarray, "Nv Nd"] = np.zeros(shape, dtype=np.uint32)
    for i, device in enumerate(devices):
        for j, density in enumerate(densities):
            mask = (npz["device"] == device) & (npz["density"] == density)
            summary[i, j] = np.sum(bugs[mask], axis=0)
            runs[i, j] = np.sum(mask)
    return summary, runs, devices, densities, reentrant


def heisenness_ci(
    key: random.PRNGKeyArray,
    alpha: Float[Array, "Nv Dd"], beta: Float[Array, "Nv Dd"],
    delta: Float[Array, "Nd"], 
    q: Float[Array, "Nq"], samples: int = 10000, correction: int = 2
) -> tuple[Float[Array, "Nq"], Float[Array, "Nq"], Float[Array, "Nq"]]:
    """Get quantiles for the heisen-ness of a bug."""
    keys = jnp.array(random.split(key, alpha.size)).reshape((*alpha.shape, -1))

    def _beta_rvs(k, a, b):
        return random.beta(k, a, b, shape=(samples,))

    Xd2_samples = vmap(vmap(_beta_rvs))(keys, alpha, beta)
    X_samples: Float[Array, "Nv Nd S"] = (
        Xd2_samples / delta[None, :, None]**(correction))
    X_mean = jnp.mean(X_samples.reshape(-1, samples), axis=0)

    H_net = jnp.std(X_samples.reshape(-1, samples), axis=0) / X_mean
    H_device = jnp.std(jnp.mean(X_samples, axis=1), axis=0) / X_mean
    H_density = jnp.std(jnp.mean(X_samples, axis=0), axis=0) / X_mean
    return (
        jnp.quantile(H_net, q),
        jnp.quantile(H_device, q),
        jnp.quantile(H_density, q))


def _main(args):
    summary, runs, devices, densities, reentrant = _get_summary_matrix(args.path)

    # Uniform prior
    prior_a = 1.0
    prior_b = 1.0
    # Remove 0 instrumentation data
    K = summary[:, 1:, :]
    n = runs[:, 1:]
    # Target quantiles
    q = jnp.array(args.quantile)

    alpha: UInt[Array, "Nv Nd Nb"] = prior_a + K
    beta: UInt[Array, "Nv Nd Nb"] = prior_b + n[:, :, None] - K
    delta: UInt[Array, "Nd"] = densities[1:] / 100

    key = random.PRNGKey(args.seed)
    keys = random.split(key, alpha.shape[-1])

    # Ordinary enumeration to make sure we don't run out of memory
    ci = []
    _iter = zip(keys, jnp.rollaxis(alpha, 2), jnp.rollaxis(beta, 2), reentrant)
    for k, a, b, r in _iter:
        ci.append(heisenness_ci(
            k, a, b, delta=delta, samples=args.samples,
            q=q, correction=1 if r else 2))

    ci_net, ci_device, ci_density = list(zip(*ci))
    np.savez(
        args.out, ci=np.array(ci_net),
        ci_device=np.array(ci_device), ci_density=np.array(ci_density),
        K=summary[:, 1:, :], runs=runs, reentrant=reentrant,
        device=devices, density=densities[1:], q=np.array(args.quantile))
