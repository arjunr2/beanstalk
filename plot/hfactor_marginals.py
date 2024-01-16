"""Heisen Factor vs Observability."""

from matplotlib import pyplot as plt
import numpy as np
import os


npz = {
    k: np.load(os.path.join("summary", k))
    for k in os.listdir("summary")
}
lower = 0
median = 2

by_device = []
by_density = []
F = []
for k, v in npz.items():
    by_device.append(
        np.mean(np.minimum(1.0, v['X'][:, median]), axis=2)[:, :21])
    by_density.append(np.mean(np.minimum(1.0, v['X'][:, median]), axis=1))
    F.append(v['F'][:, median])

F = np.concatenate(F, axis=0)
order = np.argsort(F)


def _show(ax, chunks):
    mat = np.concatenate(chunks, axis=0)
    normalized = np.nan_to_num(mat / np.nanmax(mat, axis=1)[:, None])
    return ax.imshow(normalized[order].T, aspect='auto')


fig, axs = plt.subplots(2, 1, figsize=(15, 5))
im1 = _show(axs[0], by_device)
im2 = _show(axs[1], by_density)
axs[0].set_title("Devices", loc='left')
axs[1].set_title("Instrumentation Densities", loc='left')
axs[1].set_xlabel(
    r"Bugs sorted by Increasing Heisen Factor $\longrightarrow$", loc='left')
axs[0].set_ylabel(
    r"$\longleftarrow$ Cluster Devices $\longrightarrow$", loc='center')
axs[1].set_ylabel(
    r"$\longleftarrow$ Increasing Density", loc='top')

for ax in axs:
    ax.set_yticks([])
    ax.set_xticks([])

fig.tight_layout()

fig.subplots_adjust(right=0.94)
cbar_ax = fig.add_axes([0.96, 0.075, 0.01, 0.85])
fig.colorbar(im1, cax=cbar_ax)


fig.savefig("figures/hfactor_marginals.pdf")
