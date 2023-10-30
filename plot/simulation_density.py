"""Plot density simulation."""

import numpy as np
from matplotlib import pyplot as plt
from jaxtyping import Num
from matplotlib.ticker import PercentFormatter


def _plot_ci(
    ax, x: Num[np.ndarray, "X"], y: Num[np.ndarray, "X S"], label: str,
    eps: float = 5, **kwargs
) -> None:
    y = y.astype(float)
    y[y < 0] = np.nan

    lower, upper = np.percentile(y, [eps, 100 - eps], axis=1)
    middle = np.mean(y, axis=1)

    yerr = [np.maximum(0.0, middle - lower), np.maximum(0.0, upper - middle)]

    ax.errorbar(x, middle, yerr=yerr, label=label, capsize=5, **kwargs)


beanstalk = np.load("simulations/density.npz")
baseline = np.load("simulations/baseline.npz")
benchmarks = list(beanstalk.keys())
x = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

fig, axs = plt.subplots(1, 8, figsize=(16, 3))
for ax, benchmark in zip(axs.reshape(-1), benchmarks):
    _plot_ci(
        ax, x, beanstalk[benchmark],
        label='Beanstalk', marker='D', linestyle='-', color='C0')
    ax.grid()
    ax.xaxis.set_major_formatter(PercentFormatter())
    ax.set_title(benchmark.replace(".npz", "").replace("-", "_"))
    ax.axhline(
        np.mean(baseline[benchmark][5]),
        color='C1', linestyle='--', label='Baseline', linewidth=2.0)

axs[0].set_ylabel("Number of Bugs Found $\longrightarrow$", loc='bottom')
axs[0].set_xlabel(
    "Maximum Allowed Instrumentation Density $\longrightarrow$", loc='left')
axs[-1].set_yticks([3, 4, 5, 6])
fig.tight_layout(h_pad=0.0, w_pad=-1.5)
axs[-1].legend(
    ncols=2, loc='upper right', frameon=False, bbox_to_anchor=(1.05, -0.1))

fig.savefig("figures/simulation_density.pdf")
