"""Detectability Profile Poster (slide version)."""

from matplotlib import pyplot as plt
import numpy as np
import os


npz = {
    k: np.load(os.path.join("summary", k))
    for k in os.listdir("summary")
}
lower = 2

rows = []
F = []
for k, v in npz.items():
    for i in range(v['K'].shape[-1]):
        if np.sum(v['K'][:, :, i]) > 10:
            rows.append(v['K'][:, :, i] / v['n'])
            F.append(v['F'][i, lower])

F = np.array(F)
order = np.argsort(F)

fig, axs = plt.subplots(8, 17, figsize=(16, 8))
for i, (idx, ax) in enumerate(zip(order, axs.reshape(-1))):
    ax.imshow(np.nan_to_num(
        rows[idx], nan=0.0, posinf=0.0, neginf=0.0), aspect='auto')
    ax.text(
        0.04, 0.02, "$F={:.2f}$".format(F[idx]), ha='left', va='bottom',
        color='white', transform=ax.transAxes)
    ax.text(
        0.02, 0.96, "#{:03}".format(i + 1), ha='left', va='top',
        color='white', transform=ax.transAxes)

for ax in axs.reshape(-1):
    ax.set_xticks([])
    ax.set_yticks([])

for ax in axs[-1, -8:]:
    for d in ["top", "right", "bottom", "left"]:
        ax.spines[d].set_visible(False)

fig.tight_layout(h_pad=0.3, w_pad=-0.2)
axs[-1, 0].set_xlabel(
    "Increasing Instrumentation Density $\longrightarrow$", loc='left')
axs[-1, 0].set_ylabel(
    "Different Devices $\longrightarrow$", loc='bottom')
axs[-1, -1].set_xlabel(
    "$\longleftarrow$ Decreasing Instrumentation Density", loc='right')
axs[0, 0].set_ylabel("$\longleftarrow$ Different Devices", loc='top')
fig.savefig("figures/poster2.pdf", bbox_inches='tight')
