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
    "Device\nDiversity Only": "simulations/abl_device.npz",
    "Instrumentation \nDiversity Only": "simulations/abl_density.npz",
    "Homogeneous\nBaseline": "simulations/baseline.npz"
}
data = {k: np.load(v) for k, v in methods.items()}

benchmarks = sorted(data['Beanstalk'].keys(), key=lambda x: names[x.replace(".npz", "").replace("-", "_")])
budgets = [1, 2, 5, 10, 15, 30, 60]

# Regular 3x3 grid: 8 benchmark panels fill cells 0-7; the last cell (bottom
# right) is left empty for the legend.
fig, axs = plt.subplots(3, 3, figsize=(9.5, 8))
axs_list = axs.reshape(-1)
ticks = [1, 2, 5, 15, 60]
x = np.log(budgets)

for ax, benchmark in zip(axs_list, benchmarks):
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
    ax.set_title(names[benchmark.replace(".npz", "").replace("-", "_")], fontsize=14)

# Global axis labels anchored to the bottom-left panel (now at the grid corner).
axs_list[6].set_ylabel("Number of Bugs Found $\longrightarrow$", loc='bottom', fontsize=13)
axs_list[6].set_xlabel(
    "Total Compute Budget (minutes) $\longrightarrow$", loc='left', fontsize=13)

# Use the empty bottom-right cell for the legend.
legend_ax = axs_list[8]
legend_ax.axis('off')
handles, labels = axs_list[0].get_legend_handles_labels()
legend_ax.legend(handles, labels, loc='center left', ncols=1, frameon=False, fontsize=13,
                 labelspacing=1.0)

fig.tight_layout(h_pad=0.8, w_pad=0.8)

fig.savefig("figures/simulation_budget_3grid.pdf", bbox_inches='tight', pad_inches=0.05)
