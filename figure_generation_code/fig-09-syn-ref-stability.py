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
    with open(f'../data/numerics/stability/N4000_syn_ref_randnrn_stability.pkl', 'rb') as file:
        stab_num_data = pickle.load(file)
        for k in stab_num_data.keys():
            if 'kappa' in k:
                numerics_data[k] = stab_num_data[k]

    f, axs = plt.subplots(ncols=2, nrows=2, figsize=(8 * pu.cm, 6.5 * pu.cm), dpi=150)
    ax0a = axs[0, 0]
    ax0b = axs[0, 1]
    ax1 = axs[1, 0]
    ax2 = axs[1, 1]

    ###########################################################################
    # (1) PLOT 2-D HEAT MAPS
    ###########################################################################
    cmp = 'pink'

    m = ax0a.imshow(np.flip(thry_data['mean_kappa_acc'], axis=0), cmap=cmp, origin='lower', vmin=0,
                    vmax=np.nanmax(thry_data['mean_kappa_acc']))
    cb = plt.colorbar(m, ax=ax0a)
    cb.ax.tick_params(labelsize=pu.fs1, width=0.5, pad=2)
    cb.set_ticks([0, 1, 2])
    cb.set_ticklabels(['0', '1', '2'])

    m = ax0b.imshow(np.flip(thry_data['mean_kappa_inacc'], axis=0), cmap=cmp, origin='lower', vmin=0,
                    vmax=np.nanmax(thry_data['mean_kappa_inacc']))
    cb = plt.colorbar(m, ax=ax0b)
    cb.ax.tick_params(labelsize=pu.fs1, width=0.5, pad=2)
    cb.set_ticks([0, 1, 2])
    cb.set_ticklabels(['0', '1', '2'])

    ylabs = [r'Nrn gating, $1$$-$$a$', r'Nrn gating, $1$$-$$a$']
    for i, ax in enumerate([ax0a, ax0b]):
        ax.set_xlim([0, 99])
        ax.set_ylim([0, 99])
        ax.set_xticks([0., 49, 99])
        ax.set_xticklabels(['1', '50', '100'], fontsize=pu.fs1)
        ax.set_yticks([0., 49, 99])
        ax.set_yticklabels(['0.0', '0.5', '1.0'], fontsize=pu.fs1)
        ax.tick_params(axis='both', width=0.5, length=3, pad=1)
        ax.set_xlabel(r'Nr. contexts, $s$', fontsize=pu.fs2)
        ax.set_ylabel(ylabs[i], fontsize=pu.fs2)

    ax0a.set_title('Mean accessible\n stability,'r' $\bar{\kappa}_{\text{acc}}$', fontsize=pu.fs2)
    ax0b.set_title('Mean inaccessible\n stability,'r' $\bar{\kappa}_{\text{inacc}}$', fontsize=pu.fs2)

    # formatting
    for ax in [ax1, ax2]:  # , ax3, ax4]:
        ax.set_yticks([0, 1, 2, 3])
        ax.set_yticklabels(['0', '1', '2', '3'], fontsize=pu.fs1)
        ax.set_ylabel('Mean stabilities,\n' r'$\bar{\kappa}_{\text{acc}}$, $\bar{\kappa}_{\text{inacc}}$',
                      fontsize=pu.fs2)
        ax.set_ylim([0, 3])
        ax.tick_params(axis='both', width=0.5, length=3, pad=1)

    ax1.set_xticks([1, 50, 100])
    ax1.set_xticklabels(['1', '50', '100'], fontsize=pu.fs1)
    ax1.set_xlabel(r'Nr. contexts, $s$', fontsize=pu.fs2)
    ax1.set_xlim([0, 100])

    ax2.set_xticks([0, 0.5, 1.0])
    ax2.set_xticklabels(['0.0', '0.5', '1.0'], fontsize=pu.fs1)
    ax2.set_xlabel(r'Nrn. gating, $1$$-$$a$', fontsize=pu.fs2)
    ax2.set_xlim([0, 1])

    f.subplots_adjust(wspace=1, hspace=0.)
    sns.despine()
    f.tight_layout()

    ###########################################################################
    # (2) PLOT 1-D SLICES WITH NUMERICAL COMPARISON
    ###########################################################################

    # avals and svals
    chosen_a = [0.15, 0.45, 0.85, 1.0]
    alphvals = [0.15, 0.45, 0.85, 1.0]
    chosen_s = [1, 2, 10, 50, 100]

    # theory data
    for i, a in enumerate(chosen_a):
        ind = np.where(np.abs(thry_data['avals']-a) < 1e-3)[0][0]
        print('a = ', a, thry_data['avals'][ind])
        ax1.plot(thry_data['svals'], thry_data['mean_kappa_acc'][ind, :], c='black',
                 linestyle='-', linewidth=0.5, label=f'a={a}', alpha=alphvals[i])
        ind = np.where(np.abs(numerics_data['avals'] - a) < 1e-3)[0][0]
        ax1.plot(numerics_data['svals'], numerics_data['kappa_acc'][ind, :], '.', c='black',
                 markersize=2, alpha=alphvals[i], clip_on=False)
    for i, a in enumerate(chosen_a):
        ind = np.where(np.abs(thry_data['avals'] - a) < 1e-3)[0][0]
        ax1.plot(thry_data['svals'], thry_data['mean_kappa_inacc'][ind, :], '--', c='black', linewidth=0.5,
                 label=' ', alpha=alphvals[i])
        ind = np.where(np.abs(numerics_data['avals'] - a) < 1e-3)[0][0]
        ax1.plot(numerics_data['svals'], numerics_data['kappa_inacc'][ind, :], 'o', c='black', alpha=alphvals[i],
                 markersize=2, markerfacecolor='none', markeredgewidth=0.5, clip_on=False)

    # numerics
    for i, s in enumerate(chosen_s):
        ind = np.where(np.abs(thry_data['svals'] - s) < 1e-1)[0][0]
        print('s = ', s, thry_data['svals'][ind])
        ax2.plot(1-thry_data['avals'], thry_data['mean_kappa_acc'][:, ind], c=pu.synref_clrs5[i], linewidth=0.5,
                 label=f's={s}', linestyle='-')
        ind = np.where(np.abs(numerics_data['svals'] - s) < 1e-1)[0][0]
        ax2.plot(1 - numerics_data['avals'], numerics_data['kappa_acc'][:, ind], '.', c=pu.synref_clrs5[i],
                 markersize=2)

    for i, s in enumerate(chosen_s):
        if i == 0:
            alpha = 0.
        else:  # if i > 0:
            alpha = 1.0
        ind = np.where(np.abs(thry_data['svals'] - s) < 1e-1)[0][0]
        ax2.plot(1-thry_data['avals'], thry_data['mean_kappa_inacc'][:, ind], '--', c=pu.synref_clrs5[i],
                 linewidth=0.5, label=' ', alpha=alpha)
        ind = np.where(np.abs(numerics_data['svals'] - s) < 1e-1)[0][0]
        ax2.plot(1 - numerics_data['avals'], numerics_data['kappa_inacc'][:, ind], 'o', c=pu.synref_clrs5[i],
                 markersize=2, markerfacecolor='none', markeredgewidth=0.5, alpha=alpha)

    ax0a.text(-50, 120, '(a)', c='black', fontsize=8)
    ax0b.text(-50, 120, '(b)', c='black', fontsize=8)
    ax1.text(-0.4, 1.125, '(c)', c='black', fontsize=8, transform=ax1.transAxes)
    ax2.text(-0.4, 1.125, '(d)', c='black', fontsize=8, transform=ax2.transAxes)

    ###########################################################################
    # (3) EXPANDED LEGEND
    ###########################################################################
    ax1.legend(ncol=2, frameon=False, handlelength=1.2, fontsize=pu.fs1, loc='upper right',
               bbox_to_anchor=(0.95, -0.6), markerfirst=False, columnspacing=0.5, handletextpad=1)
    ax2.legend(ncol=2, frameon=False, handlelength=1.2, fontsize=pu.fs1, loc='upper right',
               bbox_to_anchor=(0.95, -0.6), markerfirst=False, columnspacing=0.5, handletextpad=1)

    # extra text
    xoff = 0.05
    xoff2 = 0.225
    yoff = -1.78
    alphvals = [0, 0.15, 0.45, 0.85, 1.0]
    for i in range(5):
        if i > 0:
            ax1.plot([xoff2], 0.33*np.array([yoff - i * 0.36]), c='black', marker='o', linestyle='None', markersize=2,
                     alpha=alphvals[i], clip_on=False, markerfacecolor='none',
                     markeredgewidth=0.5, transform=ax1.transAxes)
        ax1.plot([xoff], 0.33*np.array([yoff - i * 0.36]), c='black', marker='.', linestyle='None', markersize=2,
                 alpha=alphvals[i], clip_on=False, transform=ax1.transAxes)
    ax1.text(-0.01, -1.8*0.33, 'acc.', fontsize=pu.fs1, color='black', clip_on=False, transform=ax1.transAxes)
    ax1.text(0.15, -1.8*0.33, 'inacc.', fontsize=pu.fs1, color='black', clip_on=False, transform=ax1.transAxes)
    ax1.text(0.6, -1.8*0.33, 'acc.', fontsize=pu.fs1, color='black', clip_on=False, transform=ax1.transAxes)
    ax1.text(0.775, -1.8*0.33, 'inacc.', fontsize=pu.fs1, color='black', clip_on=False, transform=ax1.transAxes)
    ax1.text(0.0, -1.5*0.33, 'simulation', fontsize=pu.fs1, color='black', clip_on=False, transform=ax1.transAxes)
    ax1.text(0.65, -1.5*0.33, 'theory', fontsize=pu.fs1, color='black', clip_on=False, transform=ax1.transAxes)

    xoff = 0.1
    xoff2 = 0.275
    yoff = -2.125
    for i in range(5):
        if i > 0:
            ax2.plot([xoff2], 0.33 * np.array([yoff - i * 0.36]), c=pu.synref_clrs5[i], marker='o', linestyle='None',
                     markersize=2,
                     alpha=0.75, clip_on=False, markerfacecolor='none', markeredgewidth=0.5, transform=ax2.transAxes)
        ax2.plot([xoff], 0.33 * np.array([yoff - i * 0.36]), c=pu.synref_clrs5[i], marker='.', linestyle='None',
                 markersize=2, alpha=0.75,
                 clip_on=False, transform=ax2.transAxes)
    ax2.text(0.04, -1.8 * 0.33, 'acc.', fontsize=pu.fs1, color='black', clip_on=False, transform=ax2.transAxes)
    ax2.text(0.2, -1.8 * 0.33, 'inacc.', fontsize=pu.fs1, color='black', clip_on=False, transform=ax2.transAxes)
    ax2.text(0.6, -1.8 * 0.33, 'acc.', fontsize=pu.fs1, color='black', clip_on=False, transform=ax2.transAxes)
    ax2.text(0.775, -1.8 * 0.33, 'inacc.', fontsize=pu.fs1, color='black', clip_on=False, transform=ax2.transAxes)
    ax2.text(0.05, -1.5 * 0.33, 'simulation', fontsize=pu.fs1, color='black', clip_on=False, transform=ax2.transAxes)
    ax2.text(0.65, -1.5 * 0.33, 'theory', fontsize=pu.fs1, color='black', clip_on=False, transform=ax2.transAxes)

    if save_pdf:
        f.savefig(f"{pu.fig_path}/prx_fig_09.pdf", bbox_inches='tight', pad_inches=0)
    if show_plot:
        plt.show()


if __name__ == '__main__':
    main(save_pdf=True, show_plot=True)
