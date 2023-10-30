"""Instrumentation Overhead."""

import numpy as np
from matplotlib import pyplot as plt
import os
from jaxtyping import UInt, Float
from matplotlib.ticker import PercentFormatter


def _get_runtime(path):
    npz = np.load(path)
    devices: UInt[np.ndarray, "Nv"] = np.sort(np.unique(npz["device"]))
    densities: UInt[np.ndarray, "Nd"] = np.sort(np.unique(npz["density"]))
    t: Float[np.ndarray, "Nv Nd"] = np.zeros(
        (devices.shape[0], densities.shape[0]), dtype=np.float32)

    for i, device in enumerate(devices):
        for j, density in enumerate(densities):
            mask = (npz["device"] == device) & (npz["density"] == density)
            t[i, j] = np.mean(npz['t'][mask])

    return t, devices, densities


base = "data/beanstalk"
runtimes = {p: _get_runtime(os.path.join(base, p)) for p in os.listdir(base)}

fig, ax = plt.subplots(1, 1, figsize=(4, 2.5))
densities = np.arange(21) * 5

stack = []
for k, (t, devices, densities) in runtimes.items():
    mask = (devices != 18) & (devices != 17) & (devices != 20)
    stack.append((t / t[:, 0][:, None])[mask])
stack: Float[np.ndarray, "benchmark device density"] = np.array(stack)

Y = np.exp(np.mean(np.log(stack), axis=0))
ax.errorbar(
    densities, np.mean(Y, axis=0),
    yerr=np.std(Y, axis=0, ddof=1) / np.sqrt(Y.shape[1]) * 2,
    capsize=2, marker='.')
ax.grid()
ax.xaxis.set_major_formatter(PercentFormatter())
ax.yaxis.set_major_formatter(lambda x, _: "{}x".format(int(x)))
ax.axhline(1.0, color='black', linestyle='--')
ax.set_xlabel("Instrumentation Density")
ax.set_ylabel("Runtime Overhead")
fig.tight_layout()
fig.savefig("figures/overhead.pdf")
