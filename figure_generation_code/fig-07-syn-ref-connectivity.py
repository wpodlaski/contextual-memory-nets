import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plot_utils as pu
import pickle


def main(save_pdf=False, show_plot=True):

    # load data
    with open('../data/theory/synaptic_refinement_random_neuronal_theory.pkl', 'rb') as file:
        thry_data = pickle.load(file)
    with open(f'../data/numerics/capacity/Ncxt4000_syn_ref_randnrn_capacity.pkl', 'rb') as file:
        numerics_data = pickle.load(file)

    f = plt.figure(figsize=(8 * pu.cm, 3.9 * pu.cm), dpi=150)
    gs = f.add_gridspec(6, 2, width_ratios=[1.5, 1])
    ax1 = f.add_subplot(gs[:3, 1])
    ax2 = f.add_subplot(gs[3:, 1])
    ax3 = f.add_subplot(gs[2:, 0])

    ###########################################################################
    # (1) PLOT 2-D HEAT MAP
    ###########################################################################
    cmp = 'pink'
    m = ax3.imshow(1 - thry_data['mean_c'], cmap=cmp, origin='lower', vmin=0, vmax=1.0)
    cb = plt.colorbar(m, ax=ax3)
    cb.ax.tick_params(labelsize=pu.fs1, width=0.5, pad=2)
    cb.set_ticks([0, 0.5, 1])
    cb.set_ticklabels(['0.0', '0.5', '1.0'])

    # formatting
    ylabs = [r'Nrn gating, $1$$-$$a$']
    for i, ax in enumerate([ax3]):
        ax.set_xlim([0, 99])
        ax.set_ylim([0, 99])
        ax.set_xticks([0., 49, 99])
        ax.set_xticklabels(['1', '50', '100'], fontsize=pu.fs1)
        ax.set_yticks([0., 49, 99])
        ax.set_yticklabels(['0.0', '0.5', '1.0'], fontsize=pu.fs1)
        ax.tick_params(axis='both', width=0.5, length=3, pad=1)
        ax.set_xlabel(r'Nr. contexts, $s$', fontsize=pu.fs2)
        ax.set_ylabel(ylabs[i], fontsize=pu.fs2)

    ax3.set_title('Synaptic gating with\n refinement,'r' $1$$-$$c$', fontsize=pu.fs2)

    ###########################################################################
    # (2) PLOT 1-D SLICES WITH NUMERICAL COMPARISON
    ###########################################################################
    s = [1, 2, 10, 50, 100]
    for i in range(1, len(s)):  # don't plot for s=1
        idx = np.where(thry_data['svals'] == s[i])[0][0]
        ax2.plot(1 - thry_data['avals'], 1 - thry_data['mean_c'][:, idx], linewidth=0.5, c=pu.synref_clrs5[i],
                 label=f's={s[i]}')

        idx = np.where(numerics_data['svals'] == s[i])[0][0]
        ax2.plot(1 - numerics_data['avals'], 1 - numerics_data['cvals'][:, idx], '.', c=pu.synref_clrs5[i],
                 markersize=2)

    a = [0.15, 0.45, 0.65, 1.0]
    clr = 'black'
    alphaval = [0.15, 0.45, 0.65, 1.0]
    for i in range(len(a)):
        idx = np.where(np.abs(thry_data['avals'] - a[i]) < 1e-4)[0][0]
        ax1.plot(thry_data['svals'], 1 - thry_data['mean_c'][idx, :], linewidth=0.5, c=clr, alpha=alphaval[i],
                 label=f'a={a[i]}')

        idx = np.where(np.abs(numerics_data['avals'] - a[i]) < 1e-4)[0][0]
        ax1.plot(numerics_data['svals'], 1 - numerics_data['cvals'][idx, :], '.', c=clr,
                 markersize=2, alpha=alphaval[i], clip_on=False)

    for ax in [ax1, ax2]:
        ax.tick_params(axis='both', width=0.5, length=3, pad=1)
        ax.set_yticks([0, 0.5, 1.0])
        ax.set_yticklabels(['0.0', '0.5', '1.0'], fontsize=pu.fs1)
        ax.set_ylim([0, 0.55])
    ax1.set_ylabel('Synaptic gating with\n refinement,'r' $1$$-$$c$', fontsize=pu.fs2)
    ax1.yaxis.set_label_coords(-.25, -.3)
    ax1.set_xticks([1, 50, 100])
    ax1.set_xticklabels(['1', '50', '100'], fontsize=pu.fs1)
    ax1.set_xlabel(r'Nr. contexts, $s$', fontsize=pu.fs2)
    ax1.set_xlim([0, 100])
    ax2.set_xticks([0, 0.5, 1.0])
    ax2.set_xticklabels(['0.0', '0.5', '1.0'], fontsize=pu.fs1)
    ax2.set_xlabel(r'Neuronal gating, $1$$-$$a$', fontsize=pu.fs2)
    ax2.set_xlim([0, 1])

    ###########################################################################
    # (3) EXTENDED LEGEND
    ###########################################################################
    ax1.legend(ncol=2, frameon=False, handlelength=1.2, fontsize=pu.fs1, loc='upper right',
               bbox_to_anchor=(-0.65, -2.6), markerfirst=True, columnspacing=2.5, handletextpad=1)
    ax2.legend(ncol=2, frameon=False, handlelength=1.2, fontsize=pu.fs1, loc='upper right',
               bbox_to_anchor=(0.95, -0.975), markerfirst=True, columnspacing=2.5, handletextpad=1)

    ax2.text(-2, -0.55, 'sim.', fontsize=pu.fs1, color='black', clip_on=False)
    ax2.text(-1.8, -0.55, 'theory', fontsize=pu.fs1, color='black', clip_on=False)

    ax2.plot([-1.925], [-0.65], c='black', marker='.', linestyle='None', markersize=2, alpha=0.15, clip_on=False)
    ax2.plot([-1.925], [-0.785], c='black', marker='.', linestyle='None', markersize=2, alpha=0.45, clip_on=False)
    ax2.plot([-1.25], [-0.65], c='black', marker='.', linestyle='None', markersize=2, alpha=0.65, clip_on=False)
    ax2.plot([-1.25], [-0.785], c='black', marker='.', linestyle='None', markersize=2, alpha=1.0, clip_on=False)

    ax2.plot([0.375], [-0.65], c=pu.synref_clrs5[3], marker='.', linestyle='None', markersize=2, clip_on=False)
    ax2.plot([0.375], [-0.785], c=pu.synref_clrs5[4], marker='.', linestyle='None', markersize=2, clip_on=False)
    ax2.plot([-0.175], [-0.65], c=pu.synref_clrs5[1], marker='.', linestyle='None', markersize=2, clip_on=False)
    ax2.plot([-0.175], [-0.785], c=pu.synref_clrs5[2], marker='.', linestyle='None', markersize=2, clip_on=False)

    ax3.text(-0.45, 1.4, '(a)', c='black', fontsize=8, transform=ax3.transAxes)
    ax3.text(1.55, 1.4, '(b)', c='black', fontsize=8, transform=ax3.transAxes)

    f.subplots_adjust(wspace=0.25, hspace=0.2)
    sns.despine()

    box = ax1.get_position()
    box.y0 = box.y0 + 0.05
    box.y1 = box.y1 - 0.05
    box.x0 = box.x0 + 0.1
    box.x1 = box.x1 + 0.1
    ax1.set_position(box)

    box = ax2.get_position()
    box.y0 = box.y0 + 0
    box.y1 = box.y1 - 0.1
    box.x0 = box.x0 + 0.1
    box.x1 = box.x1 + 0.1
    ax2.set_position(box)

    if save_pdf:
        f.savefig(f"{pu.fig_path}/prx_fig_07.pdf", bbox_inches='tight', pad_inches=0)
    if show_plot:
        plt.show()


if __name__ == '__main__':
    main(save_pdf=True, show_plot=True)
