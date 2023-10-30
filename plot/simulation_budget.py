"""Plot budget simulation."""

import numpy as np
from matplotlib import pyplot as plt
from jaxtyping import Num


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


beanstalk = np.load("simulations/beanstalk.npz")
baseline = np.load("simulations/baseline.npz")
benchmarks = list(beanstalk.keys())
budgets = [1, 2, 5, 10, 15, 30, 60]

fig, axs = plt.subplots(1, 8, figsize=(16, 3))
ticks = [1, 2, 5, 15, 60]
x = np.log(budgets)

for ax, benchmark in zip(axs.reshape(-1), benchmarks):
    _plot_ci(
        ax, x, baseline[benchmark],
        label='Baseline', marker='s', linestyle='--', color='C1')
    _plot_ci(
        ax, x, beanstalk[benchmark],
        label='Beanstalk', marker='D', linestyle='-', color='C0')
    ax.grid()
    ax.set_xticks(np.log(ticks))
    ax.set_xticklabels(ticks)
    ax.set_title(benchmark.replace(".npz", "").replace("-", "_"))

axs[0].set_yticks([10, 12, 14, 16, 18, 20, 22])
axs[-1].set_yticks(([3, 4, 5, 6]))
axs[0].set_ylabel("Number of Bugs Found $\longrightarrow$", loc='bottom')
axs[0].set_xlabel(
    "Total Compute Budget (minutes) $\longrightarrow$", loc='left')
fig.tight_layout(h_pad=0.0, w_pad=0.4)
axs[-1].legend(
    ncols=2, loc='upper right', bbox_to_anchor=(1.05, -0.1), frameon=False)

fig.savefig("figures/simulation_budget.pdf")
