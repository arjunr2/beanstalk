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
SUMMARY = "summary"  # detectability summaries used to number the poster panels


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


def baseline_caught(npz):
    """Site pairs the baseline caught at least once (union over the whole sweep)."""
    out = {}
    for name, v in npz.items():
        bugs = unpack_bugs(v)
        sites = v["sites"]
        out[name] = {
            tuple(sites[i]) for i in range(sites.shape[0]) if bugs[:, i].any()
        }
    return out


def poster_numbers(base=SUMMARY, K_threshold=0):
    """Map each (benchmark, site pair) to its poster panel number and F value.

    Mirrors plot/poster.py exactly: iterate summaries in os.listdir order, keep
    bugs with detection count above K_threshold, then rank by increasing F. The
    panel number printed as "#NNN" on the poster is that rank (1-based).
    """
    npz = {k: np.load(os.path.join(base, k)) for k in os.listdir(base)}
    F, bench, pair = [], [], []
    for k, v in npz.items():
        for i in range(v["K"].shape[-1]):
            if np.sum(v["K"][:, :, i]) > K_threshold:
                F.append(v["F"][i])
                bench.append(os.path.splitext(k)[0])
                pair.append(tuple(v["sites"][i]))
    F = np.array(F)
    out = {}
    for rank, idx in enumerate(np.argsort(F), start=1):
        out[(bench[idx], pair[idx])] = (rank, float(F[idx]))
    return out


def print_missing_table(beanstalk_pairs, caught, poster):
    """Print bugs found by beanstalk but missed by the baseline, with poster #s."""
    rows = []
    for name, pairs in beanstalk_pairs.items():
        for p in pairs - caught.get(name, set()):
            num, f = poster.get((name, p), (None, float("nan")))
            rows.append((num if num is not None else 1 << 30, name, p, f))
    rows.sort()

    total = sum(len(v) for v in beanstalk_pairs.values())
    print(f"\nBugs found by Beanstalk but NOT the baseline: "
          f"{len(rows)} / {total}\n")
    print(f"{'poster#':>7}  {'benchmark':<16} {'sites':<16} {'F':>7}")
    print(f"{'-' * 7}  {'-' * 16} {'-' * 16} {'-' * 7}")
    for num, name, p, f in rows:
        label = f"#{num:03d}" if num < (1 << 30) else "n/a"
        print(f"{label:>7}  {names[name]:<16} {str(tuple(p)):<16} {f:>7.2f}")


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

major_fontsize = 14
minor_fontsize = 13

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
axbig.set_ylabel("# of Bugs Detected", fontsize=major_fontsize)
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

fig.tight_layout(w_pad=0.0)
fig.savefig("figures/delay_sweep_bugs.pdf", bbox_inches="tight")

# Summary: which beanstalk-found bugs the baseline never caught, mapped to the
# poster panel numbers (plot/poster.py) via increasing detectability factor F.
print_missing_table(load_beanstalk(), baseline_caught(npz), poster_numbers())
