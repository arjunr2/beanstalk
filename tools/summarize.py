"""Calculate summary statistics."""

import numpy as np
from jax import random, vmap
from jax import numpy as jnp

from jaxtyping import UInt, Bool, Array, Float


def _parse(p):
    p.add_argument("-p", "--path", help="Path to benchmark in dataset.")
    p.add_argument("-o", "--out", help="Output file.")
    p.add_argument("-s", "--seed", help="Random seed.", type=int, default=42)
    p.add_argument(
        "--quantile", type=float, nargs='+',
        default=[0.05, 0.25, 0.5, 0.75, 0.95],
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
):
    """Get quantiles for the heisen-ness of a bug."""
    keys = jnp.array(random.split(key, alpha.size)).reshape((*alpha.shape, -1))

    def _beta_rvs(k, a, b):
        return random.beta(k, a, b, shape=(samples,))

    # Sample X_tilde from beta posterior
    X_tilde: Float[Array, "Nv Nd S"] = vmap(vmap(_beta_rvs))(keys, alpha, beta)
    # Compute X from X_tilde
    X: Float[Array, "Nv Nd S"] = (
        X_tilde / delta[None, :, None]**(correction))
    # X_tilde is conditioned on the constraint that 0 <= X <= 1; apply that
    # here.
    X = X.at[X > 1].set(jnp.nan)

    X_max = jnp.nanmax(X.reshape(-1, samples), axis=0)
    X_mean = jnp.nanmean(X.reshape(-1, samples), axis=0)
    heisen_factor = jnp.nanstd(X.reshape(-1, samples), axis=0) / X_mean
    return (
        jnp.nanquantile(X, q, axis=2),
        jnp.nanquantile(X_max, q),
        jnp.quantile(heisen_factor, q))


def _main(args):
    summary, n, devices, delta, reentrant = _get_summary_matrix(args.path)

    # Uniform prior
    prior_a = 0.1
    prior_b = 10.0
    # Remove 0 instrumentation data
    K = summary[:, 1:, :]
    n = n[:, 1:]
    delta = jnp.array(delta[1:] / 100)
    # Target quantiles
    q = jnp.array(args.quantile)

    alpha: UInt[np.ndarray, "Nv Nd Nb"] = prior_a + K
    beta: UInt[np.ndarray, "Nv Nd Nb"] = prior_b + n[:, :, None] - K

    key = random.PRNGKey(args.seed)
    keys = random.split(key, alpha.shape[-1])

    # Ordinary enumeration to make sure we don't run out of memory
    ci = []
    _iter = zip(keys, jnp.rollaxis(alpha, 2), jnp.rollaxis(beta, 2), reentrant)
    for k, a, b, r in _iter:
        ci.append(heisenness_ci(
            k, a, b, delta=delta, samples=args.samples,
            q=q, correction=1 if r else 2))

    X, X_max, F = list(zip(*ci))
    np.savez(
        args.out, F=np.array(F),
        X=np.array(X), X_max=jnp.array(X_max),
        K=K, n=n, reentrant=reentrant,
        device=devices, delta=delta, q=np.array(args.quantile))
