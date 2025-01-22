"""Instrumentation density vs detectability of an example benchmark."""

import numpy as np
from matplotlib import pyplot as plt


def format_detectability_label(y):
    label = "0"
    if y > 0.99:
        label = "100"
    else:
        y_perc = y * 100
        if y_perc < 1 and y_perc != 0:
            label = "{:.2g}".format(y_perc)[1:4]
        else:
            label = "{:.2g}".format(y_perc)[:4]
    return label

def _plot(ax, path):

    npz = np.load(path)
    mask = (
        (npz['device'] == 33) | (npz['device'] == 34) | (npz['device'] == 35))
    observations = np.sum(npz['K'][mask], axis=0)
    runs = np.sum(npz['n'][mask][:, :, None], axis=0)

    Y = observations / runs
    Y_normed = Y / np.max(Y, axis=0)[None, :]

    order = np.argsort(-np.sum(Y, axis=0))
    # Omit first 5 bugs for space
    order = order[np.max(Y, axis=0)[order] > 0][5:]

    im = ax.imshow(Y_normed[:, order], aspect='auto')
    for (j, i), y in np.ndenumerate(Y[:, order]):
        label = format_detectability_label(y)
        ax.text(i, j, label, ha='center', va='center', fontsize=9, color='purple', fontweight=800)
    return im


fig, ax = plt.subplots(1, 1, figsize=(9.5, 4.0))
im = _plot(ax, "summary/lfq.npz")
ax.set_xticks([])
ax.set_yticks(np.arange(10) * 2 + 1)
ax.set_yticklabels(["{}%".format(x + 10) for x in np.arange(10) * 10])
ax.set_ylabel(
    r"$\longleftarrow$ Increasing Instrumentation Density",
    loc='top', fontsize=11)
ax.set_xlabel(
    r"Bugs sorted by Decreasing Detectability $\longrightarrow$",
    loc='left', fontsize=11)
fig.tight_layout()
fig.savefig("figures/example.pdf")
