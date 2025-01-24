"""Detectability Profile Poster (paper version)."""

import os

import numpy as np
from matplotlib import pyplot as plt

# Number of detections below which we classify the bug as an outlier
K_threshold = 0
start_idx = 0

npz = {
    k: np.load(os.path.join("summary", k))
    for k in os.listdir("summary")
}

acronym = {
    "thread": "fib",
    "thread_lock": "fbl",
    "comp-opt-bug": "cob",
    "comp-unopt-bug": "cub",
    "loop-antidep": "ado",
    "input-dep": "ipd",
    "indirect": "ida",
    "lfq": "lfq"
}

rows = []
F = []
benchmarks = []
for k, v in npz.items():
    for i in range(v['K'].shape[-1]):
        if np.sum(v['K'][:, :, i]) > K_threshold:
            rows.append(v['K'][:, :, i] / v['n'])
            F.append(v['F'][i])
            benchmarks.append(k)

F = np.array(F)
order = np.argsort(F)
bench_acs = [acronym[benchmarks[x][:-4]] for x in order]

fig, axs = plt.subplots(11, 10, figsize=(9.5, 10.5))
for i, (idx, ax) in enumerate(zip(order[start_idx:], axs.reshape(-1))):
    ax.imshow(np.nan_to_num(
        rows[idx], nan=0.0, posinf=0.0, neginf=0.0), aspect='auto')
    ax.text(
        0.04, 0.02, "$F={:.2f}$".format(F[idx]), ha='left', va='bottom',
        color='white', transform=ax.transAxes)
    ax.text(
        0.02, 0.96, "#{:03}".format(i + 1), ha='left', va='top',
        color='white', transform=ax.transAxes)
    ax.text(
        0.98, 0.96, "{}".format(bench_acs[i]), ha='right', va='top',
        color='white', fontweight='bold', transform=ax.transAxes)

for ax in axs.reshape(-1):
    ax.set_xticks([])
    ax.set_yticks([])

for ax in axs[-1, -8:]:
    for d in ["top", "right", "bottom", "left"]:
        ax.spines[d].set_visible(False)

fig.tight_layout(h_pad=0.3, w_pad=-0.2)
axs[-1, 0].set_xlabel(
    r"Increasing Instrumentation Density $\longrightarrow$",
    loc='left', fontsize=12)
axs[-1, 0].set_ylabel(
    r"Different Devices $\longrightarrow$", loc='bottom', fontsize=12)
axs[-1, -1].set_xlabel(
    r"$\longleftarrow$ Decreasing Instrumentation Density",
    loc='right', fontsize=12)
axs[0, 0].set_ylabel(
    "$\longleftarrow$ Different Devices", loc='top', fontsize=12)
fig.savefig("figures/poster.pdf", bbox_inches='tight')
