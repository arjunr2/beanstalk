"""Instrumentation Overhead."""

import numpy as np
from matplotlib import pyplot as plt
import os
from jaxtyping import UInt, Float
from matplotlib.ticker import PercentFormatter

major_fontsize = 12
minor_fontsize = 11

xticks = np.arange(6) * 20
yticks = np.arange(13)

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


def _plot_ax(ax, x, y, title: str):
    ax.errorbar(
        densities, np.mean(y, axis=0),
        yerr=np.std(y, axis=0, ddof=1) / np.sqrt(y.shape[1]) * 2,
        capsize=2, marker='.')
    ax.grid()
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
    ax.axhline(1.0, color='black', linestyle='--')


base = "data/beanstalk"
runtimes = {p[:-4]: _get_runtime(os.path.join(base, p)) for p in os.listdir(base)}

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

width = 3
subcols = len(runtimes) // 2
fig, axs = plt.subplots(2, width+subcols, figsize=(10, 4), sharex=True, sharey=True)
gs = axs[0,0].get_gridspec()

# Axes for all benchmark specific plots and aggregate plot
subaxs = axs[:, :width]
for ax in subaxs.reshape(-1):
    ax.remove()
axbig = fig.add_subplot(gs[:, :width])

densities = np.arange(21) * 5

stack = []
for i, (k, (t, devices, densities)) in enumerate(runtimes.items()):
    mask = (devices != 18) & (devices != 17) & (devices != 20)
    Y_bench = (t / t[:, 0][:, None])[mask]
    _plot_ax(axs[i//subcols, width + (i % subcols)], densities, Y_bench, title=names[k])
    stack.append(Y_bench)

stack: Float[np.ndarray, "benchmark device density"] = np.array(stack)

Y = np.exp(np.mean(np.log(stack), axis=0))
axbig.errorbar(
    densities, np.mean(Y, axis=0),
    yerr=np.std(Y, axis=0, ddof=1) / np.sqrt(Y.shape[1]) * 2,
    capsize=2, marker='.')

axbig.grid()
axbig.xaxis.set_major_formatter(PercentFormatter())
axbig.yaxis.set_major_formatter(lambda x, _: "{}x".format(int(x)))
axbig.set_yticks(yticks)
axbig.set_xticks(xticks)
axbig.axhline(1.0, color='black', linestyle='--')
axbig.set_xlabel("Instrumentation Density", fontsize=major_fontsize)
axbig.set_ylabel("Runtime Overhead", fontsize=major_fontsize)
axbig.set_title("Aggregate", fontsize=major_fontsize)
fig.tight_layout()
fig.savefig("figures/overhead.pdf")
