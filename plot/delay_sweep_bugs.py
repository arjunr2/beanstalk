"""Distinct bugs caught vs. delay, with the beanstalk count as the ceiling.

Layout mirrors hfactor_observability.py: a large "Aggregate" panel on the left
(summed over benchmarks) and one small panel per benchmark on the right. Unlike
that figure, the y-axis tick numbers are shown, since the counts are the point.
"""

import json
import os

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator

DATA = "delay-sweep/baseline"
VIOLATIONS = "data/violations.json"


def load(base=DATA):
    """Load each benchmark's aggregated .npz from the delay-sweep dataset."""
    return {
        os.path.splitext(f)[0]: np.load(os.path.join(base, f))
        for f in sorted(os.listdir(base)) if f.endswith(".npz")
    }


def unpack_bugs(v):
    """Unpack a benchmark's per-run bug bit-vectors to shape (n_runs, n_sites)."""
    n_sites = v["sites"].shape[0]
    return np.unpackbits(v["bugs"], axis=-1)[:, :n_sites].astype(bool)


def bugs_caught(npz):
    """Distinct bugs caught per delay, per benchmark (union of runs at that delay)."""
    out = {}
    for name, v in npz.items():
        bugs = unpack_bugs(v)
        delay = v["delay"]
        out[name] = {
            int(d): int(bugs[delay == d].any(axis=0).sum())
            for d in np.unique(delay)
        }
    return out


def load_beanstalk(path=VIOLATIONS):
    """Distinct violation pairs found by beanstalk, per benchmark."""
    with open(path) as f:
        data = json.load(f)
    return {name: {tuple(p) for p in pairs} for name, pairs in data.items()}


names = {
    "thread": "fibonacci",
    "thread_lock": "fibonacci-lock",
    "comp-opt-bug": "comp-opt-bug",
    "comp-unopt-bug": "comp-unopt-bug",
    "loop-antidep": "antidep1-orig",
    "input-dep": "input-dep",
    "indirect": "indirectaccess",
    "lfq": "lock-free-queue",
}

major_fontsize = 12
minor_fontsize = 11

SWEEP = "C0"     # the delay-sweep curve
CEILING = "C3"   # beanstalk ceiling line

npz = load()
caught = bugs_caught(npz)
beanstalk = {name: len(v) for name, v in load_beanstalk().items()}
delays = sorted({int(d) for c in caught.values() for d in c})


def _plot_sweep(ax, y, ceiling, title, ms=5, xlabels=False):
    """Draw one sweep curve with its beanstalk ceiling line."""
    ax.plot(delays, y, marker="o", ms=ms, color=SWEEP, label="Baseline")
    ax.axhline(ceiling, ls="--", lw=1.2, color=CEILING, label=r"Beanstalk ($\mu = 50$)")
    ax.set_xscale("log", base=2)
    ax.set_xticks(delays)  # major tick + gridline at every delay
    if xlabels:
        ax.set_xticklabels(delays, fontsize=minor_fontsize - 2, rotation=45)
    else:
        ax.set_xticklabels([])
        ax.tick_params(axis="x", length=0)  # gridlines only, no tick marks
    ax.set_ylim(0, ceiling * 1.15)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=5))
    ax.tick_params(axis="y", labelsize=minor_fontsize - 2)
    ax.set_title(title, fontsize=minor_fontsize)
    ax.grid(axis="both", which="major", alpha=0.4)


width = 3
subcols = len(npz) // 2
# Narrower aggregate: give its columns a smaller share of the figure width.
ratios = [0.72] * width + [1.0] * subcols
fig, axs = plt.subplots(2, width + subcols, figsize=(12, 4.5),
                        gridspec_kw={"width_ratios": ratios})
gs = axs[0, 0].get_gridspec()

# Big aggregate panel spans both rows of the left `width` columns.
for ax in axs[:, :width].reshape(-1):
    ax.remove()
axbig = fig.add_subplot(gs[:, :width])

# Aggregate: distinct bugs summed across benchmarks, beanstalk total as ceiling.
agg = [sum(caught[name].get(d, 0) for name in caught) for d in delays]
agg_ceiling = sum(beanstalk.values())
_plot_sweep(axbig, agg, agg_ceiling, title="Aggregate", ms=7, xlabels=True)
axbig.set_title("Aggregate", fontsize=major_fontsize)
axbig.set_ylabel("Distinct bugs caught", fontsize=major_fontsize)
axbig.set_xlabel(r"Delay Window ($\mu$)", fontsize=major_fontsize)
axbig.set_ylim(0, 120)
axbig.set_yticks(np.arange(0, 121, 15))  # a horizontal line every 15
axbig.set_xticklabels(delays, fontsize=minor_fontsize, rotation=45)
axbig.tick_params(axis="y", labelsize=minor_fontsize)
axbig.legend(loc="lower right", fontsize=minor_fontsize, framealpha=0.9)

# Per-benchmark panels on the right, ordered by display name.
benchmarks = sorted(names, key=lambda n: names[n])
for ax, name in zip(axs[:, width:].reshape(-1), benchmarks):
    y = [caught[name].get(d, 0) for d in delays]
    _plot_sweep(ax, y, beanstalk[name], title=names[name])

fig.tight_layout()
fig.savefig("figures/delay_sweep_bugs.pdf", bbox_inches="tight")
