"""Hero figure."""

from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np

z = np.load("summary/thread.npz")
x = np.load("summary/input-dep.npz")

fig = plt.figure(figsize=(10,3.2))
gs1 = fig.add_gridspec(ncols=2, nrows=1, left=0.04, right=0.48, wspace=0.03)
gs2 = fig.add_gridspec(ncols=2, nrows=1, left=0.50, right=0.94, wspace=0.03)
axs = [fig.add_subplot(v) for v in gs1] + [fig.add_subplot(v) for v in gs2]
im1 = axs[0].imshow(z['K'][:, :, 4] / z['n'])
im2 = axs[1].imshow(x['K'][:, :, -1] / x['n'])
im3 = axs[2].imshow(z['K'][:, :, 20] / z['n'])
im4 = axs[3].imshow(x['K'][:, :, 1] / x['n'])

for ax in axs:
    ax.set_xticks([])
    ax.set_yticks([])

axs[0].set_ylabel(
    "$\longleftarrow$ Different Devices $\longrightarrow$", loc='center', fontsize=13)
axs[0].set_xlabel(
    "More Intrusive Debugging $\longrightarrow$", loc='left', fontsize=13)
axs[0].text(
    0, 0, "Ordinary Bugs", ha='left', va='top', fontsize=15, color='white')
axs[2].text(
    0, 0, "Heisenbugs", ha='left', va='top', fontsize=15, color='white')


fig.tight_layout()
cbar_ax = fig.add_axes([0.96, 0.14, 0.015, 0.71])
fig.colorbar(im1, cax=cbar_ax, orientation='vertical')
cbar_ax.set_yticks([])
cbar_ax.set_ylabel(
    "Higher Detection Probability $\longrightarrow$", loc='bottom', fontsize=11)

fig.savefig("figures/hero.pdf")
