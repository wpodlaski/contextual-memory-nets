import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plot_utils as pu
import pickle

# load theoretical values for noisy Hopfield net
with open('../data/gauss_hopfield.pkl', 'rb') as file:
    gauss_data = pickle.load(file)
alpha_Delta = gauss_data['alph_Delta']
m_Delta = gauss_data['m_Delta']


# overlap information, f_m(Delta) [Eq. 11 in the paper]
def missinfo(m):
    return 0.5 * (1 + m) * np.log2(1 + m) + 0.5 * (1 - m) * np.log2(1 - m)


def main(save_pdf=False, show_plot=True):

    # load data
    with open('../data/theory/synaptic_refinement_random_neuronal_theory.pkl', 'rb') as tmp_file:
        thry_data = pickle.load(tmp_file)
    with open(f'../data/numerics/capacity/Ncxt4000_syn_ref_randnrn_capacity.pkl', 'rb') as tmp_file:
        numerics_data = pickle.load(tmp_file)

    f, axs = plt.subplots(nrows=3, ncols=4, figsize=(15 * pu.cm, 9.5 * pu.cm), dpi=150,
                          gridspec_kw={'height_ratios': [1.2, 1, 1.2]})

    ###########################################################################
    # (1) PLOT SYNAPTIC REFINEMENT ALONE (panels a,b,c,d)
    ###########################################################################
    ax0a = axs[0, 0]
    ax0b = axs[0, 1]
    ax0c = axs[0, 2]
    ax0d = axs[0, 3]

    a = [1]
    s_idxs = [np.where(np.abs(thry_data['svals'] - numerics_data['svals'][i]) < 1e-5)[0][0] for i in
              range(len(numerics_data['svals']))]
    for i in range(len(a)):
        idx = np.where(thry_data['avals'] == a[i])[0][0]
        ax0a.plot(thry_data['svals'], thry_data['mean_m'][idx, :], linewidth=0.5, c='black')
        ax0b.plot(thry_data['svals'], thry_data['mean_alpha_cxt'][idx, :], linewidth=0.5, c='black')
        ax0c.plot(thry_data['svals'], thry_data['mean_alpha'][idx, :], linewidth=0.5, c='black')
        ax0d.plot(thry_data['svals'], missinfo(thry_data['mean_m'][idx, :]) * thry_data['mean_alpha'][idx, :],
                  linewidth=0.5, c='black')

        idx = np.where(np.abs(numerics_data['avals'] - a[i]) < 1e-4)[0][0]
        m_idx = np.where(np.abs(thry_data['avals'] - a[i]) < 1e-4)[0][0]
        ax0b.plot(numerics_data['svals'], numerics_data['alpha_cxt'][idx, :], '.', c='black', markersize=2,
                  clip_on=False)
        ax0c.plot(numerics_data['svals'], numerics_data['alpha'][idx, :], '.', c='black', markersize=2, clip_on=False)
        ax0d.plot(numerics_data['svals'], (missinfo(thry_data['mean_m'][m_idx, s_idxs])
                                           * numerics_data['alpha'][idx, :]),
                  '.', c='black', markersize=2, clip_on=False)

    ax0a.axhline(y=m_Delta(np.pi**2 / 2), c='black', linestyle='--', linewidth=0.5)
    ax0b.axhline(y=alpha_Delta(np.pi ** 2 / 2), c='black', linestyle='--', linewidth=0.5)

    ax0b.legend(('theory', 'sim'), fontsize=pu.fs1, frameon=False, handlelength=1,
                loc='upper right', bbox_to_anchor=(1.1, 0.9))

    # formatting
    for ax in [ax0a, ax0b, ax0c, ax0d]:
        ax.set_xticks([0, 49, 100])
        ax.set_xticklabels(['1', '50', '100'], fontsize=pu.fs1)
        ax.set_xlabel(r'Nr. contexts, $s$', fontsize=pu.fs2)
        ax.tick_params(axis='both', width=0.5, length=3, pad=1)
        ax.set_xlim([0, 100])

    ax0a.set_yticks([0.5, 0.75, 1.0])
    ax0a.set_yticklabels(['0.5', '0.75', '1.0'], fontsize=pu.fs1)
    ax0a.set_ylabel(r'Overlap, $m^*$', fontsize=pu.fs2)
    ax0a.set_ylim([0.5, 1])
    ax0b.set_yticks([0, 0.07, 0.14])
    ax0b.set_yticklabels(['0.0', '0.07', '0.14'], fontsize=pu.fs1)
    ax0b.set_ylabel(r'Single-cxt cap., $\alpha^*_\text{cxt}$', fontsize=pu.fs2)
    ax0b.set_ylim([0, 0.14])
    ax0c.set_yticks([0, 3, 6.0])
    ax0c.set_yticklabels(['0.0', '3.0', '6.0'], fontsize=pu.fs1)
    ax0c.set_ylabel(r'Total capacity, $\alpha^*$', fontsize=pu.fs2)
    ax0c.set_ylim([0, 6.5])
    ax0d.set_yticks([0, 2.5, 5])
    ax0d.set_yticklabels(['0.0', '2.5', '5.0'], fontsize=pu.fs1)
    ax0d.set_ylabel(r'Inf. cont. per syn., $I$', fontsize=pu.fs2)
    ax0d.set_ylim([0, 5])

    ################################################################################################
    # (2) PLOT 2-D HEAT MAPS OF COMBINED RAND NRN GATING + SYNAPTIC REFINEMENT (panels e,g,i,k)
    ################################################################################################
    ax1 = axs[1, 0]
    ax2 = axs[1, 1]
    ax3 = axs[1, 2]
    ax4 = axs[1, 3]

    cmp = 'pink'

    # overlap
    m = ax1.imshow(np.flip(thry_data['mean_m'], axis=0), cmap=cmp, origin='lower', vmin=0.5, vmax=1.0)
    cb = plt.colorbar(m, ax=ax1)
    cb.ax.tick_params(labelsize=pu.fs1, width=0.5, pad=2)
    cb.set_ticks([0.5, 0.75, 1])
    cb.set_ticklabels(['0.5', '.75', '1.0'])

    # single-context capacity
    m = ax2.imshow(np.flip(thry_data['mean_alpha_cxt'], axis=0), cmap=cmp, origin='lower', vmin=0,
                   vmax=0.14)
    cb = plt.colorbar(m, ax=ax2)
    cb.ax.tick_params(labelsize=pu.fs1, width=0.5, pad=2)
    cb.set_ticks([0, 0.07, 0.14])
    cb.set_ticklabels(['0.0', '0.07', '0.14'])

    # total capacity
    m = ax3.imshow(np.flip(thry_data['mean_alpha'], axis=0), cmap=cmp, origin='lower', vmin=0, vmax=5.5)
    cb = plt.colorbar(m, ax=ax3)
    cb.ax.tick_params(labelsize=pu.fs1, width=0.5, pad=2)
    cb.set_ticks([0, 2.5, 5])
    cb.set_ticklabels(['0.0', '2.5', '5.0'])

    # info content
    a_mesh = np.tile(thry_data['avals'], (thry_data['mean_alpha'].shape[1], 1))
    m = ax4.imshow(a_mesh * missinfo(thry_data['mean_m']) * np.flip(thry_data['mean_alpha'], axis=0),
                   cmap=cmp, origin='lower', vmin=0, vmax=4.0)
    cb = plt.colorbar(m, ax=ax4)
    cb.ax.tick_params(labelsize=pu.fs1, width=0.5, pad=2)
    cb.set_ticks([0, 2, 4])
    cb.set_ticklabels(['0.0', '2', '4'])

    ###########################################################################
    # (3) PLOT 1-D SLICES WITH NUMERICAL COMPARISON (panels f,h,j,l)
    ###########################################################################
    ax5 = axs[2, 0]
    ax6 = axs[2, 1]
    ax7 = axs[2, 2]
    ax8 = axs[2, 3]

    # theory as lines
    s = [1, 2, 10, 50, 100]
    for i in range(len(s)):
        idx = np.where(thry_data['svals'] == s[i])[0][0]
        ax5.plot(1 - thry_data['avals'], thry_data['mean_m'][:, idx], linewidth=0.5, c=pu.synref_clrs5[i])
        ax6.plot(1 - thry_data['avals'], thry_data['mean_alpha_cxt'][:, idx], linewidth=0.5, c=pu.synref_clrs5[i])
        ax7.plot(1 - thry_data['avals'], thry_data['mean_alpha'][:, idx], linewidth=0.5, c=pu.synref_clrs5[i])
        ax8.plot(1 - np.insert(thry_data['avals'][:], 0, 0),
                 missinfo(np.insert(thry_data['mean_m'][:, idx], 0, 0))
                 * np.insert(thry_data['avals'][:], 0, 0)
                 * np.insert(thry_data['mean_alpha'][:, idx], 0, 0),
                 c=pu.synref_clrs5[i], markersize=2, linewidth=0.5, alpha=1)

    ax5.legend(('$s$=1', '$s$=2', '$s$=10', '$s$=50', '$s$=100'), handlelength=1, ncol=1, columnspacing=1.5,
               fontsize=pu.fs1, frameon=False, loc='upper right', bbox_to_anchor=(1., 0.75))

    # numerical data as points
    s = [1, 2, 10, 50, 100]
    a_idxs = [np.where(np.abs(thry_data['avals'] - numerics_data['avals'][i]) < 1e-5)[0][0]
              for i in range(len(numerics_data['avals']))]
    for i in range(len(s)):
        idx = np.where(numerics_data['svals'] == s[i])[0][0]
        m_idx = np.where(thry_data['svals'] == s[i])[0][0]
        ax6.plot(1 - numerics_data['avals'], numerics_data['alpha_cxt'][:, idx], '.', c=pu.synref_clrs5[i],
                 markersize=2)
        ax7.plot(1 - numerics_data['avals'], numerics_data['alpha'][:, idx], '.', c=pu.synref_clrs5[i], markersize=2)
        ax8.plot(1 - numerics_data['avals'], missinfo(thry_data['m'][a_idxs, m_idx])
                 * numerics_data['avals'] * numerics_data['alpha'][:, idx], '.', c=pu.synref_clrs5[i], markersize=2)

    # formatting for all panels
    for i, ax in enumerate([ax1, ax2, ax3, ax4]):
        ax.set_xlim([0, 99])
        ax.set_ylim([0, 99])
        ax.set_xticks([0., 49, 99])
        ax.set_xticklabels(['1', '50', '100'], fontsize=pu.fs1)
        ax.set_yticks([0., 49, 99])
        ax.set_yticklabels(['0.0', '0.5', '1.0'], fontsize=pu.fs1)
        ax.tick_params(axis='both', width=0.5, length=3, pad=1)
        ax.set_xlabel(r'Nr. contexts, $s$', fontsize=pu.fs2)
        ax.set_ylabel(r'Nrn gating, $1$$-$$a$', fontsize=pu.fs2)

    for ax in [ax5, ax6, ax7, ax8]:
        ax.set_xlim([0, 1])
        ax.set_xticks([0., 0.5, 1.0])
        ax.set_xticklabels(['0.0', '0.5', '1.0'], fontsize=pu.fs1)
        ax.tick_params(axis='both', width=0.5, length=3, pad=1)
        ax.set_xlabel(r'Neuronal gating, $1$$-$$a$', fontsize=pu.fs2)
    ax5.set_yticks([0.5, 0.75, 1.0])
    ax5.set_yticklabels(['0.5', '0.75', '1.0'], fontsize=pu.fs1)
    ax6.set_yticks([0., 0.07, 0.14])
    ax6.set_yticklabels(['0.0', '0.07', '0.14'], fontsize=pu.fs1)
    ax7.set_yticks([0., 3, 6])
    ax7.set_yticklabels(['0.0', '3.0', '6.0'], fontsize=pu.fs1)
    ax8.set_yticks([0., 2.5, 5])
    ax8.set_yticklabels(['0.0', '2.5', '5.0'], fontsize=pu.fs1)
    ax5.set_ylim([0.5, 1])
    ax6.set_ylim([0, 0.14])
    ax7.set_ylim([0, 6])
    ax8.set_ylim([0, 5])
    ax5.set_ylabel(r'Overlap, $m^*$', fontsize=pu.fs2)
    ax6.set_ylabel(r'Single-cxt cap., $\alpha^*_\text{cxt}$', fontsize=pu.fs2)
    ax7.set_ylabel(r'Total capacity, $\alpha^*$', fontsize=pu.fs2)
    ax8.set_ylabel(r'Inf. cont. per syn., $I$', fontsize=pu.fs2)
    ax0a.set_title(r'Overlap, $m^*$', fontsize=pu.fs2)
    ax0b.set_title(r'Single-cxt capacity, $\alpha^*_\text{cxt}$', fontsize=pu.fs2)
    ax0c.set_title(r'Total capacity, $\alpha^*$', fontsize=pu.fs2)
    ax0d.set_title(r'Inf. content per syn., $I$', fontsize=pu.fs2)

    f.subplots_adjust(wspace=0, hspace=0.)
    sns.despine()
    f.tight_layout()

    ax0a.text(-0.4, 1.15, '(a)', c='black', fontsize=8, transform=ax0a.transAxes)
    ax0b.text(-0.4, 1.15, '(b)', c='black', fontsize=8, transform=ax0b.transAxes)
    ax0c.text(-0.4, 1.15, '(c)', c='black', fontsize=8, transform=ax0c.transAxes)
    ax0d.text(-0.4, 1.15, '(d)', c='black', fontsize=8, transform=ax0d.transAxes)
    ax1.text(-0.54, 1.1, '(e)', c='black', fontsize=8, transform=ax1.transAxes)
    ax2.text(-0.54, 1.1, '(g)', c='black', fontsize=8, transform=ax2.transAxes)
    ax3.text(-0.54, 1.1, '(i)', c='black', fontsize=8, transform=ax3.transAxes)
    ax4.text(-0.54, 1.1, '(k)', c='black', fontsize=8, transform=ax4.transAxes)
    ax5.text(-0.4, 1.1, '(f)', c='black', fontsize=8, transform=ax5.transAxes)
    ax7.text(-0.4, 1.1, '(j)', c='black', fontsize=8, transform=ax7.transAxes)
    ax6.text(-0.4, 1.1, '(h)', c='black', fontsize=8, transform=ax6.transAxes)
    ax8.text(-0.4, 1.1, '(l)', c='black', fontsize=8, transform=ax8.transAxes)

    for ax in [ax1, ax2, ax3, ax4]:
        box = ax.get_position()
        box.x0 = box.x0 - 0.011
        box.x1 = box.x1 - 0.011
        ax.set_position(box)

    if save_pdf:
        f.savefig(f"{pu.fig_path}/prx_fig_08.pdf", bbox_inches='tight', pad_inches=0)
    if show_plot:
        plt.show()


if __name__ == '__main__':
    main(save_pdf=True, show_plot=True)
