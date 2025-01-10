import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plot_utils as pu
import pickle


# overlap information, f_m(Delta) [Eq. 11 in the paper]
def missinfo(m):
    return 0.5 * (1 + m) * np.log2(1 + m) + 0.5 * (1 - m) * np.log2(1 - m)


def main(save_pdf=False, show_plot=True):

    f, axs = plt.subplots(nrows=2, ncols=4, figsize=(15 * pu.cm, 6.5 * pu.cm), dpi=150,
                          gridspec_kw={'height_ratios': [1, 1.2]})
    ax1 = axs[0, 0]
    ax2 = axs[0, 1]
    ax3 = axs[0, 2]
    ax4 = axs[0, 3]

    # load data
    with open('../data/theory/random_neuronal_gating_theory.pkl', 'rb') as tmp_file:
        nrn_theory_data = pickle.load(tmp_file)
    with open(f'../data/numerics/capacity/Ncxt4000_rand_nrn_gating_capacity.pkl', 'rb') as tmp_file:
        numerics_data = pickle.load(tmp_file)

    ###########################################################################
    # (1) PLOT 2-D HEAT MAPS
    ###########################################################################
    cmp = 'pink'

    # overlap, m
    m = ax1.imshow(np.flip(nrn_theory_data['mean_m'], axis=0), cmap=cmp, origin='lower', vmin=0.6, vmax=1.)
    cb = plt.colorbar(m, ax=ax1)
    cb.ax.tick_params(labelsize=pu.fs1, width=0.5, pad=2)
    cb.set_ticks([0.6, 0.8, 1.])
    cb.set_ticklabels(['0.6', '0.8', '1.0'])

    # single-context capacity, alpha_cxt
    m = ax2.imshow(np.flip(nrn_theory_data['mean_alpha_cxt'], axis=0), cmap=cmp, origin='lower', vmin=0, vmax=0.14)
    cb = plt.colorbar(m, ax=ax2)
    cb.ax.tick_params(labelsize=pu.fs1, width=0.5, pad=2)
    cb.set_ticks([0, 0.07, 0.14])
    cb.set_ticklabels(['0.0', '.07', '0.14'])

    # total capacity, alpha
    m = ax3.imshow(np.flip(nrn_theory_data['mean_alpha'], axis=0), cmap=cmp, origin='lower', vmin=0,
                   vmax=np.max(nrn_theory_data['mean_alpha']))
    cb = plt.colorbar(m, ax=ax3)
    cb.ax.tick_params(labelsize=pu.fs1, width=0.5, pad=2)
    cb.set_ticks([0, 0.5, 1.0])
    cb.set_ticklabels(['0.0', '0.5', '1.0'])

    # info content, I
    info_content = (np.tile(1 - nrn_theory_data['avals'], (100, 1)).T
                    * np.flip(nrn_theory_data['mean_alpha'], axis=0)
                    * missinfo(np.flip(nrn_theory_data['mean_m'], axis=0)))
    m = ax4.imshow(info_content, cmap=cmp, origin='lower', vmin=0, vmax=np.max(info_content))
    cb = plt.colorbar(m, ax=ax4)
    cb.ax.tick_params(labelsize=pu.fs1, width=0.5, pad=2)
    cb.set_ticks([0, 0.08, 0.16])
    cb.set_ticklabels(['0.0', '0.08', '0.16'])

    # formatting
    ylabs = [r'Nrn gating, $1$$-$$a$', r'Nrn gating, $1$$-$$a$', r'Nrn gating, $1$$-$$a$', r'Nrn gating, $1$$-$$a$',
             r'Syn gating, $1$$-$$c$', r'Syn gating, $1$$-$$c$', r'Syn gating, $1$$-$$c$', r'Syn gating, $1$$-$$c$']
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
    ax1.set_title(r'Overlap, $m^*$', fontsize=pu.fs2)
    ax2.set_title(r'Single-cxt capacity, $\alpha^*_\text{cxt}$', fontsize=pu.fs2)
    ax3.set_title(r'Total capacity, $\alpha^*$', fontsize=pu.fs2)
    ax4.set_title(r'Info. content, $I^*$', fontsize=pu.fs2)

    ###########################################################################
    # (2) PLOT 1-D SLICES WITH NUMERICAL COMPARISON
    ###########################################################################
    ax5 = axs[1, 0]
    ax6 = axs[1, 1]
    ax7 = axs[1, 2]
    ax8 = axs[1, 3]

    # plot theory as lines
    s = [1, 2, 10, 100]
    avals = nrn_theory_data['avals']
    svals = nrn_theory_data['svals']
    for i in range(len(s)):
        idx = np.where(svals == s[i])[0][0]
        ax5.plot(1 - avals, nrn_theory_data['mean_m'][:, idx], linewidth=0.5, c=pu.nrn_clrs4[i])
        ax6.plot(1 - avals, nrn_theory_data['mean_alpha_cxt'][:, idx], linewidth=0.5, c=pu.nrn_clrs4[i])
        ax7.plot(1 - avals, nrn_theory_data['mean_alpha'][:, idx], linewidth=0.5, c=pu.nrn_clrs4[i])
        ax8.plot(1 - avals, avals * nrn_theory_data['mean_alpha'][:, idx] * missinfo(nrn_theory_data['mean_m'][:, idx]),
                 linewidth=0.5, c=pu.nrn_clrs4[i])

    # plot numerics as points
    avals = numerics_data['avals']
    svals = numerics_data['svals']
    a_idxs = [np.where(np.abs(nrn_theory_data['avals'] - avals[i]) < 1e-5)[0][0] for i in range(len(avals))]
    for i in range(len(s)):
        idx = np.where(svals == s[i])[0][0]
        m_idx = np.where(nrn_theory_data['svals'] == s[i])[0][0]
        ax6.plot(1 - avals, numerics_data['alpha_cxt'][:, idx], '.', linewidth=0.5, c=pu.nrn_clrs4[i], markersize=2)
        ax7.plot(1 - avals, numerics_data['alpha'][:, idx], '.', linewidth=0.5, c=pu.nrn_clrs4[i], markersize=2)
        ax8.plot(1 - avals, avals * numerics_data['alpha'][:, idx] * missinfo(nrn_theory_data['m'][a_idxs, m_idx]), '.',
                 linewidth=0.5, c=pu.nrn_clrs4[i], markersize=2)

    # formatting
    ax5.legend(('$s$=1', '$s$=2', '$s$=10', '$s$=100'), handlelength=1.5,
               fontsize=pu.fs1, frameon=False, loc='upper right', bbox_to_anchor=(1, 0.75))
    ax6.text(0.2, 0.15, 'sim', fontsize=pu.fs1)  # 0.13
    ax6.text(0.63, 0.15, 'theory', fontsize=pu.fs1)
    ax6.plot([0.12], [0.153], '.', c='black', markersize=2)
    ax6.plot([0.45, 0.58], [0.153, 0.153], '-', c='black', linewidth=0.5)

    for ax in [ax5, ax6, ax7, ax8]:
        ax.set_xlim([0, 1])
        ax.set_xticks([0., 0.5, 1.0])
        ax.set_xticklabels(['0.0', '0.5', '1.0'], fontsize=pu.fs1)
        ax.tick_params(axis='both', width=0.5, length=3, pad=1)
    ax5.set_yticks([0., 0.5, 1.0])
    ax5.set_yticklabels(['0.0', '0.5', '1.0'], fontsize=pu.fs1)
    ax5.set_ylim([0, 1])
    ax6.set_yticks([0., 0.07, 0.14])
    ax6.set_yticklabels(['0.0', '0.07', '0.14'], fontsize=pu.fs1)
    ax6.set_ylim([0, 0.16])
    ax7.set_yticks([0., 0.5, 1.0])
    ax7.set_yticklabels(['0.0', '0.5', '1.0'], fontsize=pu.fs1)
    ax7.set_ylim([0, 1.1])
    ax8.set_yticks([0., 0.08, 0.16])
    ax8.set_yticklabels(['0.0', '0.08', '0.16'], fontsize=pu.fs1)
    ax8.set_ylim([0, 0.18])

    for ax in [ax5, ax6, ax7, ax8]:
        ax.set_xlabel(r'Neuronal gating, $1$$-$$a$', fontsize=pu.fs2)
    ax5.set_ylabel(r'Overlap, $m^*$', fontsize=pu.fs2)
    ax6.set_ylabel(r'Single-cxt cap., $\alpha^*_\text{cxt}$', fontsize=pu.fs2)
    ax7.set_ylabel(r'Total capacity, $\alpha^*$', fontsize=pu.fs2)
    ax8.set_ylabel(r'Info. content, $I^*$', fontsize=pu.fs2)

    f.subplots_adjust(wspace=0, hspace=0.)
    sns.despine()
    f.tight_layout()

    ax1.text(-55, 120, '(a)', c='black', fontsize=8)
    ax2.text(-55, 120, '(c)', c='black', fontsize=8)
    ax3.text(-55, 120, '(e)', c='black', fontsize=8)
    ax4.text(-55, 120, '(g)', c='black', fontsize=8)
    ax5.text(-0.4, 1.1, '(b)', c='black', fontsize=8, transform=ax5.transAxes)
    ax6.text(-0.4, 1.1, '(d)', c='black', fontsize=8, transform=ax6.transAxes)
    ax7.text(-0.4, 1.1, '(f)', c='black', fontsize=8, transform=ax7.transAxes)
    ax8.text(-0.4, 1.1, '(h)', c='black', fontsize=8, transform=ax8.transAxes)

    for ax in [ax1, ax2, ax3, ax4]:
        box = ax.get_position()
        box.x0 = box.x0 - 0.011
        box.x1 = box.x1 - 0.011
        ax.set_position(box)

    if save_pdf:
        f.savefig(f"{pu.fig_path}/prx_fig_03.pdf", bbox_inches='tight', pad_inches=0)
    if show_plot:
        plt.show()


if __name__ == '__main__':
    main(save_pdf=True, show_plot=True)
