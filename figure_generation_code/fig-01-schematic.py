import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plot_utils as pu
from scipy.stats import norm


def main(save_pdf=False, show_plot=True):

    # set up figure panels
    f = plt.figure(figsize=(8 * pu.cm, 5 * pu.cm), dpi=100)
    grid = f.add_gridspec(2, 12, height_ratios=[1, 1.5])
    ax1 = f.add_subplot(grid[0, 0:4])
    ax2 = f.add_subplot(grid[0, 4:8])
    ax3 = f.add_subplot(grid[0, 8:12])
    ax4 = f.add_subplot(grid[1, :6])
    ax5 = f.add_subplot(grid[1, 8:])
    ax2ds = [ax1, ax2, ax3]

    # panel (a): weight matrices with gating illustrated
    new_cmap = pu.truncate_colormap(plt.get_cmap('binary'), 0., 0.6)
    W = np.random.rand(6, 6)
    W[np.triu_indices(6, 1)] = W[np.tril_indices(6, -1)]
    W -= np.diag(np.diag(W))
    m = None
    for i in range(3):
        m = ax2ds[i].imshow(W, vmin=0, vmax=1, cmap=new_cmap)
        ax2ds[i].set_xticks([])
        ax2ds[i].set_yticks([])
    cb = plt.colorbar(m, ax=ax2ds[2])
    cb.ax.tick_params(labelsize=pu.fs1, width=0.5, pad=2)
    cb.set_ticks([])

    ldx = -0.25
    ldx2 = 0.25
    ldy = -0.25
    ldy2 = 0.25
    ax2ds[0].plot([1, 1], [0 + ldy, 5 + ldy2], c='red', linewidth=1.25)
    ax2ds[0].plot([3, 3], [0 + ldy, 5 + ldy2], c='red', linewidth=1.25)
    ax2ds[0].plot([0 + ldx, 5 + ldx2], [1, 1], c='red', linewidth=1.25)
    ax2ds[0].plot([0 + ldx, 5 + ldx2], [3, 3], c='red', linewidth=1.25)

    dx = -0.15
    dy = 0.25
    ax2ds[1].text(0 + dx, 2 + dy, 'x', c='red', fontsize=7)
    ax2ds[1].text(2 + dx, 0 + dy, 'x', c='red', fontsize=7)
    ax2ds[1].text(1 + dx, 4 + dy, 'x', c='red', fontsize=7)
    ax2ds[1].text(4 + dx, 1 + dy, 'x', c='red', fontsize=7)
    ax2ds[1].text(2 + dx, 5 + dy, 'x', c='red', fontsize=7)
    ax2ds[1].text(5 + dx, 2 + dy, 'x', c='red', fontsize=7)

    ax2ds[2].plot([1, 1], [0 + ldy, 5 + ldy2], c='red', linewidth=1.25)
    ax2ds[2].plot([3, 3], [0 + ldy, 5 + ldy2], c='red', linewidth=1.25)
    ax2ds[2].plot([0 + ldx, 5 + ldx2], [1, 1], c='red', linewidth=1.25)
    ax2ds[2].plot([0 + ldx, 5 + ldx2], [3, 3], c='red', linewidth=1.25)
    ax2ds[2].text(0 + dx, 2 + dy, 'x', c='red', fontsize=7)
    ax2ds[2].text(2 + dx, 0 + dy, 'x', c='red', fontsize=7)
    ax2ds[2].text(1 + dx, 4 + dy, 'x', c='red', fontsize=7)
    ax2ds[2].text(4 + dx, 1 + dy, 'x', c='red', fontsize=7)
    ax2ds[2].text(2 + dx, 5 + dy, 'x', c='red', fontsize=7)
    ax2ds[2].text(5 + dx, 2 + dy, 'x', c='red', fontsize=7)

    # panel (b): 1-d energy landscapes
    x = np.linspace(0, 1, 501)
    centers = np.linspace(0.1, 0.9, 9)
    weights = -0.15 * np.abs(np.random.normal(size=(3, centers.shape[0])))
    z = 35.
    idxs = [[4, 6], [5, 8], [2, 3, 7]]
    offsets = [2, 1, -0.5]
    for i in range(3):
        weights[i, idxs[i]] -= 1
        y = np.zeros_like(x)
        for j in range(centers.shape[0]):
            y += weights[i, j] * norm(loc=centers[j], scale=0.025).pdf(x) / z
        ax4.plot(x, offsets[i] + y, linewidth=0.75, color=pu.context_cmap[i])

    # panel (c): accessibility vs nr. contexts
    s = 1 + np.arange(100)
    ax5.plot(s, 1. / s, color='black', alpha=0.8, linewidth=1, label=r'$1/s$')

    # figure formatting
    ax5.set_xticks([1, 50, 100])
    ax5.set_xticklabels(['1', '50', '100'], fontsize=pu.fs1)
    ax5.set_yticks([0.0, 0.5, 1.0])
    ax5.set_yticklabels(['0.0', '0.5', '1.0'], fontsize=pu.fs1)
    ax5.set_ylabel('Prop. of accessible\n memories', fontsize=pu.fs2)
    ax5.set_xlabel(r'Nr. contexts, $s$', fontsize=pu.fs2)
    ax5.tick_params(axis='both', width=0.5, length=3, pad=1)
    ax5.set_ylim([-0.02, 1.1])

    ax4.set_xticks([])
    ax4.set_yticks([])
    sns.despine(ax=ax4, left=True, bottom=True)
    sns.despine(ax=ax5)
    f.tight_layout()

    ax1.text(-0.35, 1.2, '(a)', c='black', fontsize=8, transform=ax1.transAxes)
    ax4.text(-0.1825, 1.12, '(b)', c='black', fontsize=8, transform=ax4.transAxes)
    ax5.text(-0.575, 1.075, '(c)', c='black', fontsize=8, transform=ax5.transAxes)

    ax1.text(-0.5, -1.15, 'neuronal gating', c='black', fontsize=6)
    ax2.text(-0.4, -1.15, 'synaptic gating', c='black', fontsize=6)
    ax3.text(-0.6, -1.15, 'combined gating', c='black', fontsize=6)
    ax4.text(-0.02, 0.825, 'context 1', c=pu.context_cmap[0], fontsize=5, transform=ax4.transAxes)
    ax4.text(-0.02, 0.525, 'context 2', c=pu.context_cmap[1], fontsize=5, transform=ax4.transAxes)
    ax4.text(0.5, 0.315, '...', fontsize=10, c='black', clip_on=False, rotation=90, transform=ax4.transAxes)
    ax4.text(-0.02, 0.085, r'context $s$', c=pu.context_cmap[2], fontsize=5, transform=ax4.transAxes)
    ax4.text(-0.05, -0.2, 'network state', c='black', fontsize=6, transform=ax4.transAxes)
    ax4.arrow(0.45, -0.18, 0.1, 0, width=0.0001, head_width=0.02, head_length=0.01,
              color='black', transform=ax4.transAxes, clip_on=False)
    ax4.text(-0.15, -0.1, 'energy', c='black', fontsize=6, transform=ax4.transAxes, rotation=90)
    ax4.arrow(-0.12, 0.225, 0, 0.1, width=0.0002, head_width=0.02, head_length=0.03,
              color='black', transform=ax4.transAxes, clip_on=False)

    ax3.text(6.5, 1.25, 'strong', fontsize=5, rotation=-90)
    ax3.text(6.5, 5.35, 'weak', fontsize=5, rotation=-90)

    box = ax4.get_position()
    box.x0 = box.x0 + 0.04
    box.x1 = box.x1 - 0.01
    box.y0 = box.y0 - 0.07
    box.y1 = box.y1 - 0.07
    ax4.set_position(box)

    box = ax5.get_position()
    box.y0 = box.y0 - 0.05
    box.y1 = box.y1 - 0.05
    ax5.set_position(box)

    if save_pdf:
        f.savefig(f"{pu.fig_path}/prx_fig_01.pdf", bbox_inches='tight', pad_inches=0.18)
    if show_plot:
        plt.show()


if __name__ == '__main__':
    main(save_pdf=True, show_plot=True)
