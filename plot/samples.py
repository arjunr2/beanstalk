"""Detectability Profile Poster (paper version)."""

from matplotlib import pyplot as plt
import numpy as np
import os


npz = {
    k: np.load(os.path.join("summary", k))
    for k in os.listdir("summary")
}
lower = 0

rows = []
F = []
for k, v in npz.items():
    for i in range(v['K'].shape[-1]):
        if np.sum(v['K'][:, :, i]) > 10:
            rows.append(v['K'][:, :, i] / v['n'])
            F.append(v['F'][i])

F = np.array(F)
order = np.argsort(F)


samples = {
    "normal": [1, 3, 4, 5],
    "onedevice": [93, 117, 121, 134],
    "similar": [11, 37, 41, 50],
    "random": [99, 101, 104, 119]
}


for desc, idxs in samples.items():
    fig, axs = plt.subplots(2, 2, figsize=(2.4, 2.4))
    for ii, ax in zip(idxs, axs.reshape(-1)):
        ax.imshow(np.nan_to_num(
            rows[order[ii - 1]], nan=0.0, posinf=0.0, neginf=0.0
        ), aspect='auto')
        ax.text(
            0.04, 0.02, "$F={:.2f}$".format(F[order[ii - 1]]),
            ha='left', va='bottom', color='white', transform=ax.transAxes)

    for ax in axs.reshape(-1):
        ax.set_xticks([])
        ax.set_yticks([])

    fig.tight_layout(h_pad=0.3, w_pad=0.3)
    axs[-1, 0].set_xlabel(
        "Instrumentation Density $\longrightarrow$", loc='left')
    axs[1, 0].set_ylabel(
        " " * 3 +
        "$\longleftarrow$ Different Devices $\longrightarrow$", loc='bottom')
    fig.savefig("figures/sample_{}.pdf".format(desc), bbox_inches='tight')
