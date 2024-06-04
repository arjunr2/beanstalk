"""Detectability Profile Poster (printed version)."""

from matplotlib import pyplot as plt
import numpy as np
import os


npz = {
    k: np.load(os.path.join("summary", k))
    for k in os.listdir("summary")
}
median = 2

rows = []
F = []
for k, v in npz.items():
    for i in range(v['K'].shape[-1]):
        rows.append(v['K'][:, :, i] / v['n'])
    F.append(v['F'][:, median])

F = np.concatenate(F, axis=0)
order = np.argsort(F)

fig, axs = plt.subplots(16, 12, figsize=(10, 15))
for i, (idx, ax) in enumerate(zip(order, axs.reshape(-1))):
    ax.imshow(np.nan_to_num(rows[idx], nan=0.0), aspect='auto')
    ax.text(
        1, 1, "${:.3f}$".format(F[idx]), ha='left', va='top', color='white')
    ax.set_xticks([])
    ax.set_yticks([])

fig.tight_layout(h_pad=0.3, w_pad=0.3)
fig.subplots_adjust(left=0.05, bottom=0.08, right=0.965, top=0.89)
axs[-1, 0].set_xlabel(
    "Increasing Instrumentation Density $\longrightarrow$", loc='left')
axs[-1, 0].set_ylabel(
    "Different Devices $\longrightarrow$", loc='bottom')
axs[-1, -1].set_xlabel(
    "$\longleftarrow$ Decreasing Instrumentation Density", loc='right')
axs[0, 0].set_ylabel("$\longleftarrow$ Different Devices", loc='top')
fig.text(
    0.045, 0.965, "Distribute and Conquer",
    fontsize=36, ha='left', va='top', weight='bold')
fig.text(
    0.046, 0.925, "Data Race Detection with Beanstalk",
    fontsize=24, va='top')
fig.text(
    0.05, 0.05, """Detectability profile of 192 bugs in our dataset, sorted by Heisen Factor $F$. Each plot shows the detectability of a single bug
across different devices (y-axis) and instrumentation densities (x-axis).""", va='top')

fig.savefig("figures/beanstalk_12x18.pdf")
