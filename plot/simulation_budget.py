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


names = {
    "thread": "fibonacci",
    "thread_lock": "fibonacci-lock",
    "comp_opt_bug": "comp-opt-bug",
    "comp_unopt_bug": "comp-unopt-bug",
    "loop_antidep": "antidep1-orig",
    "input_dep": "input-dep",
    "indirect": "indirectaccess",
    "lfq": "lock-free-queue"
}
methods = {
    "Beanstalk": "simulations/beanstalk.npz",
    "Device Diversity Only": "simulations/abl_device.npz",
    "Instrumentation Diversity Only": "simulations/abl_density.npz",
    "Homogenous Baseline": "simulations/baseline.npz"
}
data = {k: np.load(v) for k, v in methods.items()}

benchmarks = list(data["Beanstalk"].keys())
budgets = [1, 2, 5, 10, 15, 30, 60]

fig, axs = plt.subplots(2, 4, figsize=(12, 5))
ticks = [1, 2, 5, 15, 60]
x = np.log(budgets)

for ax, benchmark in zip(axs.reshape(-1), benchmarks):
    markers = 's^Dv'
    lines = ['-', '--', ':', '-.']
    colors = ['C0', 'C1', 'C2', 'C3']
    for (k, v), m, ls, c in zip(data.items(), markers, lines, colors):
        invalid = np.any(v[benchmark] == -1, axis=(1, 2))
        nbugs = np.sum(v[benchmark], axis=2).astype(np.float32)
        nbugs[invalid] = np.nan
        _plot_ci(ax, x, nbugs, label=k, marker=m, linestyle=ls, color=c)

    ax.grid()
    ax.set_xticks(np.log(ticks))
    ax.set_xticklabels(ticks)
    ax.set_title(names[benchmark.replace(".npz", "").replace("-", "_")])

axs[0, 0].set_yticks([10, 12, 14, 16, 18, 20, 22])
axs[0, 2].set_yticks([12, 14, 16, 18, 20, 22])
axs[1, 3].set_yticks(([3, 4, 5, 6, 7]))
axs[1, 0].set_ylabel("Number of Bugs Found $\longrightarrow$", loc='bottom')
axs[1, 0].set_xlabel(
    "Total Compute Budget (minutes) $\longrightarrow$", loc='left')
fig.tight_layout(h_pad=0.5, w_pad=0.4)
axs[-1, -1].legend(
    ncols=4, loc='upper right', bbox_to_anchor=(1.05, -0.09), frameon=False)

fig.savefig("figures/simulation_budget.pdf")
