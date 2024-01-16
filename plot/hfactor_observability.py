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
    "comp_opt_bug": "comp-opt-bug",
    "comp_unopt_bug": "comp-unopt-bug",
    "loop_antidep": "antidep1-orig",
    "input_dep": "input-dep",
    "indirect": "indirectaccess",
    "lfq": "lock-free-queue"
}

fig, ax = plt.subplots(1, 1, figsize=(6, 4))
for k, v in npz.items():
    X = v['X_max']
    y = v["F"]
    mask = np.sum(v['K'], axis=(0, 1)) > 10
    X = X[mask]
    y = y[mask]

    ax.errorbar(
        X[:, median], y[:, median],
        yerr=[y[:, median] - y[:, q1], y[:, q3] - y[:, median]],
        linestyle='', marker='.',
        label=names[k.replace('.npz', '').replace('-', '_')], linewidth=1.0)
ax.grid()
ax.set_ylabel(r"Heisen Factor")
ax.set_xlabel(r"Best Conditional Observability")
ax.legend()
ax.set_yticks([0, 5, 10, 15])
fig.tight_layout()
fig.savefig("figures/hfactor_observability.pdf")
