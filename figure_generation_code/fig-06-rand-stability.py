import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plot_utils as pu
import pickle


def main(save_pdf=False, show_plot=True):

    f, axs = plt.subplots(nrows=3, ncols=2, figsize=(8 * pu.cm, 11 * pu.cm), dpi=150,
                          gridspec_kw={'height_ratios': [1, 1, 1.1]})
    ax1 = axs[0, 0]
    ax2 = axs[1, 0]
    ax3 = axs[0, 1]
    ax4 = axs[1, 1]

    # load neuronal gating data
    with open('../data/theory/random_neuronal_gating_theory.pkl', 'rb') as tmp_file:
        nrn_theory_data = pickle.load(tmp_file)
    with open(f'../data/numerics/capacity/Ncxt4000_rand_nrn_gating_capacity.pkl', 'rb') as tmp_file:
        nrn_numerics_data = pickle.load(tmp_file)
    with open(f'../data/numerics/stability/N4000_rand_nrn_gating_stability.pkl', 'rb') as tmp_file:
        stab_num_data = pickle.load(tmp_file)
        for k in stab_num_data.keys():
            if 'kappa' in k:
                nrn_numerics_data[k] = stab_num_data[k]

    # load synaptic gating data
    with open('../data/theory/random_synaptic_gating_theory.pkl', 'rb') as tmp_file:
        syn_theory_data = pickle.load(tmp_file)
    with open(f'../data/numerics/capacity/Ncxt4000_rand_syn_gating_capacity.pkl', 'rb') as tmp_file:
        syn_numerics_data = pickle.load(tmp_file)
    with open(f'../data/numerics/stability/N4000_rand_syn_gating_stability.pkl', 'rb') as tmp_file:
        stab_num_data = pickle.load(tmp_file)
        for k in stab_num_data.keys():
            if 'kappa' in k:
                syn_numerics_data[k] = stab_num_data[k]

    # fix nan problem
    for i, key in enumerate(['mean_kappa_acc', 'mean_kappa_inacc']):
        for j in range(syn_theory_data[key].shape[1]):
            nan_ids = np.where(np.isnan(syn_theory_data[key][:, j]))[0]
            if len(nan_ids) > 0:
                for k, idx in enumerate(nan_ids[::-1]):
                    syn_theory_data[key][idx, j] = syn_theory_data[key][idx + 1, j]

    ###########################################################################
    # (1) PLOT 2-D HEAT MAPS
    ###########################################################################
    cmp = 'pink'

    # accessible stability for neuronal gating
    m = ax1.imshow(np.flip(nrn_theory_data['mean_kappa_acc'], axis=0), cmap=cmp, origin='lower', vmin=0,
                   vmax=np.nanmax(nrn_theory_data['mean_kappa_acc']))
    cb = plt.colorbar(m, ax=ax1)
    cb.ax.tick_params(labelsize=pu.fs1, width=0.5, pad=2)
    cb.set_ticks([0, 1, 2])
    cb.set_ticklabels(['0', '1', '2'])

    # accessible stability for synaptic gating
    m = ax3.imshow(np.flip(syn_theory_data['mean_kappa_acc'], axis=0), cmap=cmp, origin='lower', vmin=0,
                   vmax=np.nanmax(syn_theory_data['mean_kappa_acc']))
    cb = plt.colorbar(m, ax=ax3)
    cb.ax.tick_params(labelsize=pu.fs1, width=0.5, pad=2)
    cb.set_ticks([0, 1, 2])
    cb.set_ticklabels(['0', '1', '2'])

    # inaccessible stability for neuronal gating
    m = ax2.imshow(np.flip(nrn_theory_data['mean_kappa_inacc'], axis=0), cmap=cmp, origin='lower', vmin=0,
                   vmax=np.nanmax(nrn_theory_data['mean_kappa_inacc']))
    cb = plt.colorbar(m, ax=ax2)
    cb.ax.tick_params(labelsize=pu.fs1, width=0.5, pad=2)
    cb.set_ticks([0, 1, 2])
    cb.set_ticklabels(['0', '1', '2'])

    # inaccessible stability for synaptic gating
    m = ax4.imshow(np.flip(syn_theory_data['mean_kappa_inacc'], axis=0), cmap=cmp, origin='lower', vmin=0,
                   vmax=np.nanmax(syn_theory_data['mean_kappa_inacc']))
    cb = plt.colorbar(m, ax=ax4)
    cb.ax.tick_params(labelsize=pu.fs1, width=0.5, pad=2)
    cb.set_ticks([0, 1, 2])
    cb.set_ticklabels(['0', '1', '2'])

    # formatting
    ylabs = [r'Nrn gating, $1$$-$$a$', r'Nrn gating, $1$$-$$a$',
             r'Syn gating, $1$$-$$c$', r'Syn gating, $1$$-$$c$']
    for i, ax in enumerate([ax1, ax2, ax3, ax4]):
        ax.set_xlim([0, 99])
        ax.set_ylim([0, 99])
        ax.set_xticks([0., 49, 99])
        ax.set_xticklabels(['1', '50', '100'], fontsize=pu.fs1)
        ax.set_yticks([0., 49, 99])
        ax.set_yticklabels(['0.0', '0.5', '1.0'], fontsize=pu.fs1)
        ax.tick_params(axis='both', width=0.5, length=3, pad=1)
        ax.set_xlabel(r'Nr. contexts, $s$', fontsize=pu.fs2)
        ax.set_ylabel(ylabs[i], fontsize=pu.fs2)
    ax1.set_title('Mean accessible\n stability,'r' $\bar{\kappa}_{\text{acc}}$', fontsize=pu.fs2)
    ax2.set_title('Mean inaccessible\n stability,'r' $\bar{\kappa}_{\text{inacc}}$', fontsize=pu.fs2)
    ax3.set_title('Mean accessible\n stability,'r' $\bar{\kappa}_{\text{acc}}$', fontsize=pu.fs2)
    ax4.set_title('Mean inaccessible\n stability,'r' $\bar{\kappa}_{\text{inacc}}$', fontsize=pu.fs2)

    ###########################################################################
    # (2) PLOT 1-D SLICES WITH NUMERICAL COMPARISON
    ###########################################################################
    ax5 = axs[2, 0]
    ax6 = axs[2, 1]

    # plot theory as lines
    svals = np.array([1, 2, 10, 100])
    cvals = np.insert(syn_theory_data['cvals'], 0, 0)
    for i, s in enumerate(svals):
        idx = np.where(nrn_theory_data['svals'] == s)[0][0]
        ax5.plot(1 - nrn_theory_data['avals'], nrn_theory_data['mean_kappa_acc'][:, idx],
                 c=pu.nrn_clrs4[i], linewidth=0.4, linestyle='-', label=f's={s}')
        ax6.plot(1 - cvals, np.insert(syn_theory_data['mean_kappa_acc'][:, idx], 0, -1),
                 c=pu.syn_clrs4[i], linewidth=0.4, linestyle='-', label=f's={s}')
    for i, s in enumerate(svals):
        if i == 0:
            alpha = 0.  # don't plot inaccessible stability for s=1
        else:
            alpha = 1.
        idx = np.where(nrn_theory_data['svals'] == s)[0][0]
        ax5.plot(1 - nrn_theory_data['avals'], nrn_theory_data['mean_kappa_inacc'][:, idx],
                 c=pu.nrn_clrs4[i], linewidth=0.4, linestyle='--', label=' ', alpha=alpha)
        ax6.plot(1 - cvals, np.insert(syn_theory_data['mean_kappa_inacc'][:, idx], 0, -1),
                 c=pu.syn_clrs4[i], linewidth=0.4, linestyle='--', label=' ', alpha=alpha)

    # plot numerics as points
    for i, s in enumerate(svals):
        idx = np.where(nrn_numerics_data['svals'] == s)[0][0]
        ax5.plot(1 - nrn_numerics_data['avals'], nrn_numerics_data['kappa_acc'][:, idx], '.',
                 c=pu.nrn_clrs4[i], markersize=2)
        ax5.plot(1 - nrn_numerics_data['avals'], nrn_numerics_data['kappa_inacc'][:, idx], 'o',
                 c=pu.nrn_clrs4[i], markersize=2, markerfacecolor='none', markeredgewidth=0.5)
        idx = np.where(syn_numerics_data['svals'] == s)[0][0]
        ax6.plot(1 - syn_numerics_data['cvals'], syn_numerics_data['kappa_acc'][:, idx], '.',
                 c=pu.syn_clrs4[i], markersize=2)
        ax6.plot(1 - syn_numerics_data['cvals'], syn_numerics_data['kappa_inacc'][:, idx], 'o',
                 c=pu.syn_clrs4[i], markersize=2, markerfacecolor='none', markeredgewidth=0.5)

    # formatting
    for ax in [ax5, ax6]:  # , ax7, ax8]:
        ax.set_xlim([0, 1])
        ax.set_xticks([0., 0.5, 1.0])
        ax.set_xticklabels(['0.0', '0.5', '1.0'], fontsize=pu.fs1)
        ax.tick_params(axis='both', width=0.5, length=3, pad=1)
        ax.set_ylim([0, 3])
        ax.set_yticks([0., 1, 2, 3])
        ax.set_yticklabels(['0', '1', '2', '3'], fontsize=pu.fs1)

    for ax in [ax5]:  # , ax6]:
        ax.set_xlabel(r'Neuronal gating, $1$$-$$a$', fontsize=pu.fs2)
    for ax in [ax6]:  # , ax8]:
        ax.set_xlabel(r'Synaptic gating, $1$$-$$c$', fontsize=pu.fs2)
    ax5.set_ylabel(r'Mean stabilities', fontsize=pu.fs2)
    ax6.set_ylabel(r'Mean stabilities', fontsize=pu.fs2)

    sns.despine()
    f.tight_layout()

    ###########################################################################
    # (3) EXTENDED LEGEND
    ###########################################################################
    ax5.legend(ncol=2, frameon=False, handlelength=1.2, fontsize=pu.fs1, loc='upper right',
               bbox_to_anchor=(0.95, -0.6), markerfirst=False, columnspacing=0.5, handletextpad=1)
    ax6.legend(ncol=2, frameon=False, handlelength=1.2, fontsize=pu.fs1, loc='upper right',
               bbox_to_anchor=(0.95, -0.6), markerfirst=False, columnspacing=0.5, handletextpad=1)

    # extra text
    xoff = 0.1
    xoff2 = 0.275
    yoff = -2.125
    for i in range(4):
        if i > 0:
            ax5.plot([xoff2], [yoff - i*0.36], c=pu.nrn_clrs4[i], marker='o', linestyle='None', markersize=2, alpha=0.75,
                     clip_on=False, markerfacecolor='none', markeredgewidth=0.5)
            ax6.plot([xoff2], [yoff - i * 0.36], c=pu.syn_clrs4[i], marker='o', linestyle='None', markersize=2, alpha=0.75,
                     clip_on=False, markerfacecolor='none', markeredgewidth=0.5)
        ax5.plot([xoff], [yoff - i*0.36], c=pu.nrn_clrs4[i], marker='.', linestyle='None', markersize=2, alpha=0.75,
                 clip_on=False)
        ax6.plot([xoff], [yoff - i * 0.36], c=pu.syn_clrs4[i], marker='.', linestyle='None', markersize=2, alpha=0.75,
                 clip_on=False)
    for ax in [ax5, ax6]:
        ax.text(0.04, -1.8, 'acc.', fontsize=pu.fs1, color='black', clip_on=False)
        ax.text(0.2, -1.8, 'inacc.', fontsize=pu.fs1, color='black', clip_on=False)
        ax.text(0.6, -1.8, 'acc.', fontsize=pu.fs1, color='black', clip_on=False)
        ax.text(0.775, -1.8, 'inacc.', fontsize=pu.fs1, color='black', clip_on=False)
        ax.text(0.05, -1.5, 'simulation', fontsize=pu.fs1, color='black', clip_on=False)
        ax.text(0.65, -1.5, 'theory', fontsize=pu.fs1, color='black', clip_on=False)

    ax1.text(-43, 120, '(a)', c='black', fontsize=8)
    ax2.text(-43, 120, '(b)', c='black', fontsize=8)
    ax3.text(-43, 120, '(d)', c='black', fontsize=8)
    ax4.text(-43, 120, '(e)', c='black', fontsize=8)
    ax5.text(-0.32, 1.1, '(c)', c='black', fontsize=8, transform=ax5.transAxes)
    ax6.text(-0.32, 1.1, '(f)', c='black', fontsize=8, transform=ax6.transAxes)

    for ax in [ax1, ax2, ax3, ax4]:
        box = ax.get_position()
        box.x0 = box.x0 - 0.0155
        box.x1 = box.x1 - 0.0155
        ax.set_position(box)

    for ax in [ax2, ax4]:
        box = ax.get_position()
        box.y0 = box.y0 - 0.02
        box.y1 = box.y1 - 0.02
        ax.set_position(box)

    if save_pdf:
        f.savefig(f"{pu.fig_path}/prx_fig_06.pdf", bbox_inches='tight', pad_inches=0)
    if show_plot:
        plt.show()


if __name__ == '__main__':
    main(save_pdf=True, show_plot=True)
