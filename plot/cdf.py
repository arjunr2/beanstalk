import numpy as np
from matplotlib import pyplot as plt
import os


methods = {
    "Beanstalk": "simulations/beanstalk.npz",
    "Device Diversity Only": "simulations/abl_device.npz",
    "Instrumentation Diversity Only": "simulations/abl_density.npz",
    "Homogenous Baseline": "simulations/baseline.npz"
}
markers = {
    "Beanstalk": "-",
    "Device Diversity Only": "--",
    "Instrumentation Diversity Only": ":",
    "Homogenous Baseline": "-."
}


data = {}
for name, path in methods.items():
    bugs = []
    npz = np.load(path)
    for k, v in npz.items():
        valid = np.all(v != -1, axis=(1, 2))
        bugs.append(np.mean(v[valid][5], axis=0))
    data[name] = np.concatenate(bugs)


fig, axs = plt.subplots(2, 1, figsize=(6, 3.75), height_ratios=(1, 5), sharex=True)

for k, v in data.items():
    for ax in axs:
        ax.plot(
            [-1, *np.sort(v), 2], [0, *((np.arange(len(v)) + 1) / len(v)), 1],
            label=k, linestyle=markers[k])

for ax in axs:
    ax.grid()
    ax.set_xlim(-0.05, 1.05)
axs[0].set_ylim(0.92, 1.02)
axs[1].set_ylim(-0.02, 0.58)

axs[0].spines.bottom.set_visible(False)
axs[1].spines.top.set_visible(False)
for tick in axs[0].xaxis.get_major_ticks():
    tick.tick1line.set_visible(False)
    tick.tick2line.set_visible(False)
    tick.label1.set_visible(False)
    tick.label2.set_visible(False)

kwargs = dict(
    marker=[(-1, -0.5), (1, 0.5)], markersize=12,
    linestyle="none", color='k', mec='k', mew=1, clip_on=False)
axs[0].plot([0, 1], [0, 0], transform=axs[0].transAxes, **kwargs)
axs[1].plot([0, 1], [1, 1], transform=axs[1].transAxes, **kwargs)

axs[0].set_yticks([1.0])
axs[0].set_yticklabels(["1.0"])
axs[1].legend(loc='lower right')

axs[1].set_xlabel("Detection Probability")
axs[1].set_ylabel("        Cumulative Probability")

fig.tight_layout(h_pad=0.0)

sty = {
    "ha": "center", "va": "center", "fontsize": 12, "backgroundcolor": "white"}
axs[1].text(0.93, 0.58, "$(1)$", **sty)
axs[1].text(0.65, 0.45, "$(2)$", **sty)
axs[1].text(0.18, 0.39, "$(3)$", **sty)
axs[1].annotate(
    '', xy=(-0.01, 0.34), xytext=(0.36, 0.38), arrowprops={"arrowstyle": "<->"})
axs[1].annotate(
    '', xy=(0.35, 0.38), xytext=(0.97, 0.47), arrowprops={"arrowstyle": "<->"})
axs[1].annotate(
    '', xy=(0.97, 0.47), xytext=(0.97, 0.73), arrowprops={"arrowstyle": "<->"})

fig.savefig("figures/cdf.pdf")
