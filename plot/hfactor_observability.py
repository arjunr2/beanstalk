"""Heisen Factor vs Observability."""

from matplotlib import pyplot as plt
import numpy as np
import os


npz = {
    k: np.load(os.path.join("summary", k))
    for k in os.listdir("summary")
}
q1 = 0
median = 2
q3 = 4
names = {
    "thread": "fibonacci",
    "thread_lock": "fibonacci-lock",
    "comp-opt-bug": "comp-opt-bug",
    "comp-unopt-bug": "comp-unopt-bug",
    "loop-antidep": "antidep1-orig",
    "input-dep": "input-dep",
    "indirect": "indirectaccess",
    "lfq": "lock-free-queue"
}

xticks = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
yticks = [0, 2.5, 5, 7.5, 10]
major_fontsize=12
minor_fontsize=11

def _scatter_ax(ax, x, y, title: str):
    ax.scatter(x, y, marker='.', color='C0', s=48)
    ax.set_title(title, fontsize=minor_fontsize)
    ax.set_yticks(yticks)
    ax.set_xticks(xticks)
    for tick in ax.yaxis.get_major_ticks():
        tick.tick1line.set_visible(False)
        tick.tick2line.set_visible(False)
        tick.label1.set_visible(False)
        tick.label2.set_visible(False)
    for tick in ax.xaxis.get_major_ticks():
        tick.tick1line.set_visible(False)
        tick.tick2line.set_visible(False)
        tick.label1.set_visible(False)
        tick.label2.set_visible(False)
    ax.grid()

width = 3
subcols = len(npz) // 2
fig, axs = plt.subplots(2, width+subcols, figsize=(10, 4), sharey=True, sharex=True)
gs = axs[0,0].get_gridspec()

# Axes for all benchmark specific plots
subaxs = axs[:, :width]
for ax in subaxs.reshape(-1):
    ax.remove()
axbig = fig.add_subplot(gs[:, :width])


for i, (k, v) in enumerate(npz.items()):
    X = v['X_max']
    y = v["F"]
    mask = np.sum(v['K'], axis=(0, 1)) > 10
    X = X[mask]
    y = y[mask]
    _scatter_ax(axs[i//subcols, width + (i % subcols)], X[:, median], y[:, 0], title=names[k[:-4]])
    axbig.scatter(X[:, median], y[:, 0], marker='.', color='C0', s=48)

axbig.grid()
axbig.set_ylabel(r"Heisen Factor", fontsize=major_fontsize)
axbig.set_xlabel(r"Best Conditional Observability", fontsize=major_fontsize)
axbig.set_yticks(yticks)
axbig.set_xticks(xticks)
axbig.set_title(r"Aggregate", fontsize=major_fontsize)
fig.tight_layout()
fig.savefig("figures/hfactor_observability.pdf")
