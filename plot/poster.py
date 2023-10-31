"""Detectability Profile Poster."""

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

fig, axs = plt.subplots(15, 13, figsize=(10, 12))
for i, (idx, ax) in enumerate(zip(order, axs.reshape(-1))):
    ax.imshow(np.nan_to_num(rows[idx], nan=0.0), aspect='auto')
    ax.text(
        1, 1, "${:.3f}$".format(F[idx]), ha='left', va='top', color='white')

for ax in axs.reshape(-1):
    ax.set_xticks([])
    ax.set_yticks([])

for ax in axs[-1, -2:]:
    for d in ["top", "right", "bottom", "left"]:
        ax.spines[d].set_visible(False)

fig.tight_layout(h_pad=0.3, w_pad=0.3)
axs[-1, 0].set_xlabel(
    "Increasing Instrumentation Density $\longrightarrow$", loc='left')
axs[-1, 0].set_ylabel(
    "Different Devices $\longrightarrow$", loc='bottom')
axs[-1, -1].set_xlabel(
    "$\longleftarrow$ Decreasing Instrumentation Density", loc='right')
axs[0, 0].set_ylabel("$\longleftarrow$ Different Devices", loc='top')
fig.savefig("figures/poster.pdf", bbox_inches='tight')
