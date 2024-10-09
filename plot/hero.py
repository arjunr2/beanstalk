"""Hero figure."""

from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np

z = np.load("summary/thread.npz")
x = np.load("summary/input-dep.npz")

fig = plt.figure(figsize=(10,3.2))
gs1 = fig.add_gridspec(ncols=2, nrows=1, left=0.04, right=0.50, wspace=0.03)
gs2 = fig.add_gridspec(ncols=2, nrows=1, left=0.52, right=0.98, wspace=0.03)
#gs1.tight_layout(fig)
#gs2.tight_layout(fig)
axs = [fig.add_subplot(v) for v in gs1] + [fig.add_subplot(v) for v in gs2]
#fig, axs = plt.subplots(1, 4, figsize=(10, 3.5))
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
fig.subplots_adjust(top=1)
cbar_ax = fig.add_axes([0.63, 0.08, 0.35, 0.04])
fig.colorbar(im1, cax=cbar_ax, orientation='horizontal')
cbar_ax.set_xticks([])
cbar_ax.set_xlabel(
    "Higher Probability of Detection $\longrightarrow$", loc='left', fontsize=12)

fig.savefig("figures/hero.pdf")
