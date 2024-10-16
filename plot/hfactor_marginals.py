"""Heisen Factor vs Observability."""

import os

import numpy as np
from matplotlib import pyplot as plt

npz = {
    k: np.load(os.path.join("summary", k))
    for k in os.listdir("summary")
}

by_device = []
by_density = []
F = []
for k, v in npz.items():
    by_device.append(
        np.mean(np.minimum(1.0, v['X']), axis=2)[:, :21])
    by_density.append(np.mean(np.minimum(1.0, v['X']), axis=1))
    F.append(v['F'])

F = np.concatenate(F, axis=0)
order = np.argsort(F)


def _show(ax, chunks):
    mat = np.concatenate(chunks, axis=0)
    normalized = np.nan_to_num(mat / np.nanmax(mat, axis=1)[:, None])
    return ax.imshow(normalized[order].T, aspect='auto')


fig, axs = plt.subplots(2, 1, figsize=(12, 5))
im1 = _show(axs[0], by_device)
im2 = _show(axs[1], by_density)
axs[0].set_title("Devices", loc='left', fontsize=14)
axs[1].set_title("Instrumentation Densities", loc='left', fontsize=14)
axs[1].set_xlabel(
    r"Bugs sorted by Increasing Heisen Factor $\longrightarrow$", loc='left', fontsize=13)
axs[0].set_ylabel(
    r"$\longleftarrow$ Cluster Devices $\longrightarrow$", loc='center', fontsize=12)
axs[1].set_ylabel(
    r"$\longleftarrow$ Increasing Density", loc='top', fontsize=12)

for ax in axs:
    ax.set_yticks([])
    ax.set_xticks([])

fig.tight_layout()

fig.subplots_adjust(right=0.94)
cbar_ax = fig.add_axes((0.955, 0.075, 0.01, 0.85))
cbar_ax.tick_params(labelsize=13)
fig.colorbar(im1, cax=cbar_ax)


fig.savefig("figures/hfactor_marginals.pdf")
