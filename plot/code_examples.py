"""
    Plots to accompany code example ('thread'/'fibonacci' benchmark)
"""
""" This is the specific benchmark layout used (from code_example.c)

    #define N 20000
    // Global State
    long fibonacci[2] = {1, 0};
    int num_ct = 1;
    volatile int p = 0;

    void *fib_thread(void *tid) {
    int tnum = *(int *)tid;
    while (num_ct < N) {
        p += tnum;
        long f1 = fibonacci[1];
        fibonacci[1] = fibonacci[0];
        fibonacci[0] += f1;
        num_ct++;
    }
    return NULL;
    }

    int main() {
    // Spawn + join 2 threads
    // invoking `fib_thread(tid)`
    spawn_threads(2, fib_thread);
        return 0;
    }
"""

import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as patches


npz = np.load("summary/thread.npz")
N = npz['n']
F, N, K = npz['F'], npz['n'], npz['K']
bench_sites = npz['sites']
devices = npz['device']

# Maps the bug site (wasm instruction) to line of source code (C)
# e.g. code_map[4] corresponds to line of code with access idx 4
code_map = [0] * 4 + [
    8, 10, 10, 9, 11, 12, 12, 14, 13, 10, 10
]

# Devices to highlight in the plot (color + linestyle)
dev1 = (31, 'red', '--', 'Intel NUC 11 (x86-64 : tigerlake)')
dev2 = (19, 'magenta', (0, (3,1,1,1)), 'Intel Compute Stick (x86-64 : silvermont)')

toCodeSite = lambda a: (code_map[a[0]], code_map[a[1]])

target_sites = {
    (12, 12): 0,
    (10, 10): 1,
    (13, 14): 2,
    (7, 11): 3,
    (8, 10): 4,
    (9, 12): 5
}

site_idxs = [(0, (0, 0))] * len(target_sites)
for i, site in enumerate(bench_sites):
    st = target_sites.get(tuple(site), len(target_sites))
    if st < len(target_sites):
        site_idxs[st] = (i, site)


fig, axs = plt.subplots(2, 3, figsize=(6.5, 4.5))
for i, ((idx, site), ax) in enumerate(zip(site_idxs, axs.reshape(-1))):
    ax.imshow(np.nan_to_num(
        K[:, :, idx] / N, nan=0.0, posinf=0.0, neginf=0.0), aspect='auto')
    ax.text(
        0.02, 0.96, "{}".format(toCodeSite(site)), ha='left', va='top',
        color='white', fontweight='bold', fontsize=13, transform=ax.transAxes)
    ax.set_xticks([])
    ax.set_yticks([])

for ax in axs[-1, -8:]:
    for d in ["top", "right", "bottom", "left"]:
        ax.spines[d].set_visible(False)

fig.tight_layout(h_pad=0.4, w_pad=0.4)
axs[-1, 0].set_xlabel(
    r"Increasing Instrumentation Density $\longrightarrow$",
    loc='left', fontsize=12)
axs[-1, 0].set_ylabel(
    r"$\longleftarrow$ Different Devices $\longrightarrow$", loc='center', fontsize=12)


dev1_idx = np.where(devices == dev1[0])[0][0]
dev2_idx = np.where(devices == dev2[0])[0][0]
for ax in axs.reshape(-1):
    ax.add_patch(
        patches.Rectangle((-0.5, 0.5 + dev1_idx), 20, 1, fill=False, edgecolor=dev1[1], lw=2, ls=dev1[2], label=dev1[3])
    )
    ax.add_patch(
        patches.Rectangle((-0.5, 0.5 + dev2_idx), 20, 1, fill=False, edgecolor=dev2[1], lw=2, ls=dev2[2], label=dev2[3])
    )

axs[-1,0].legend(
    ncols=1, loc='lower left', frameon=False, bbox_to_anchor=(-0.05, -0.45), fontsize=12)

fig.savefig("figures/code_examples.pdf", bbox_inches='tight')