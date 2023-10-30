"""Instrumentation density vs detectability of an example benchmark."""

import numpy as np
from matplotlib import pyplot as plt


def _plot(ax, path):

    npz = np.load(path)
    mask = (
        (npz['device'] == 33) | (npz['device'] == 34) | (npz['device'] == 35))
    observations = np.sum(npz['K'][mask], axis=0)
    runs = np.sum(npz['n'][mask][:, :, None], axis=0)

    Y = observations / runs
    Y_normed = Y / np.max(Y, axis=0)[None, :]

    order = np.argsort(-np.sum(Y, axis=0))
    order = order[np.max(Y, axis=0)[order] > 0]

    im = ax.imshow(Y_normed[:, order], aspect='auto')
    for (j, i), label in np.ndenumerate(Y[:, order]):
        if label > 0.99:
            label = "100"
        else:
            label = "{:.2g}".format(label * 100)[:4]
        ax.text(i, j, label, ha='center', va='center', fontsize=8)
    return im


fig, ax = plt.subplots(1, 1, figsize=(6.5, 4.5))
im = _plot(ax, "summary/thread.npz")
ax.set_xticks([])
ax.set_yticks(np.arange(10) * 2 + 1)
ax.set_yticklabels(["{}%".format(x + 10) for x in np.arange(10) * 10])
ax.set_ylabel(
    r"$\longleftarrow$ Increasing Instrumentation Density",
    loc='top', fontsize=12)
ax.set_xlabel(
    r"Bugs sorted by Decreasing Detectability $\longrightarrow$",
    loc='left', fontsize=12)
fig.tight_layout()
fig.savefig("figures/example.pdf")
