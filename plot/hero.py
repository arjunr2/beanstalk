"""Hero figure."""

from matplotlib import pyplot as plt
import numpy as np

z = np.load("summary/thread.npz")

fig, axs = plt.subplots(1, 2, figsize=(6, 4))
im1 = axs[0].imshow(z['K'][:, :, 4] / z['n'])
im2 = axs[1].imshow(z['K'][:, :, 20] / z['n'])

for ax in axs:
    ax.set_xticks([])
    ax.set_yticks([])

axs[0].set_ylabel(
    "$\longleftarrow$ Different Devices $\longrightarrow$", loc='center')
axs[0].set_xlabel(
    "Higher Debugging Intensities $\longrightarrow$", loc='left')
axs[0].text(
    0, 0, "Ordinary Bug", ha='left', va='top', fontsize=14, color='white')
axs[1].text(
    0, 0, "Heisenbug", ha='left', va='top', fontsize=14, color='white')


fig.tight_layout()
fig.subplots_adjust(bottom=0.05)
cbar_ax = fig.add_axes([0.53, 0.06, 0.445, 0.04])
fig.colorbar(im1, cax=cbar_ax, orientation='horizontal')
cbar_ax.set_xticks([])
cbar_ax.set_xlabel(
    "Higher Probability of Detection $\longrightarrow$", loc='left')

fig.savefig("figures/hero.pdf")
