import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plot_utils as pu
import pickle
from matplotlib.patches import Rectangle


# load theoretical values for noisy Hopfield net
with open('../data/gauss_hopfield.pkl', 'rb') as file:
    gauss_data = pickle.load(file)
alpha_Delta = gauss_data['alph_Delta']
m_Delta = gauss_data['m_Delta']


# overlap information, f_m(Delta) [Eq. 11 in the paper]
def missinfo(m):
    return 0.5 * (1 + m) * np.log2(1 + m) + 0.5 * (1 - m) * np.log2(1 - m)


def main(save_pdf=False, show_plot=True):

    f, axs = plt.subplots(ncols=2, nrows=2, figsize=(8 * pu.cm, 6.5 * pu.cm), dpi=150)
    ax1 = axs[1, 0]
    ax2 = axs[1, 1]
    ax3 = axs[0, 1]
    ax4 = axs[0, 0]

    # load data
    with open('../data/theory/synaptic_refinement_arbitrary_theory.pkl', 'rb') as tmp_file:
        thry_data = pickle.load(tmp_file)
    with open(f'../data/numerics/capacity/Ncxt4000_syn_ref_arbitrary_capacity.pkl', 'rb') as tmp_file:
        numerics_data = pickle.load(tmp_file)
    with open(f'../data/numerics/stability/N4000_syn_ref_arbitrary_stability.pkl', 'rb') as tmp_file:
        stab_num_data = pickle.load(tmp_file)
        for k in stab_num_data.keys():
            if 'kappa' in k:
                numerics_data[k] = stab_num_data[k]

    # load data for random gaussian weights
    with open(f'../data/numerics/capacity/Ncxt4000_syn_ref_randgauss_capacity.pkl', 'rb') as tmp_file:
        randgauss_data = pickle.load(tmp_file)
    with open(f'../data/numerics/stability/N4000_syn_ref_randgauss_stability.pkl', 'rb') as tmp_file:
        stab_num_data = pickle.load(tmp_file)
        for k in stab_num_data.keys():
            if 'kappa' in k:
                randgauss_data[k] = stab_num_data[k]

    # select the data points to plot
    inds = np.where(np.in1d(numerics_data['svals'], np.array([1, 2, 4, 8, 16, 32, 64, 128])))[0]

    # panel (c): single-context capacity
    ax1.plot(thry_data['svals'][:-5] * thry_data['alpha_cxt'][:-5], thry_data['alpha_cxt'][:-5], linewidth=0.5,
             c='black')
    n_idx = np.where(numerics_data['avals'] == 1)[0][0]
    ax1.plot(numerics_data['svals'][inds] * numerics_data['alpha_cxt'][n_idx, inds],
             numerics_data['alpha_cxt'][n_idx, inds], '.', c='black', markersize=2)
    ax1.axhline(y=alpha_Delta(np.pi ** 2 / 2), c='black', linestyle=':', linewidth=0.5)

    # panel (b): synaptic gating, 1 - c
    ax3.plot(thry_data['svals'][:-5] * thry_data['alpha_cxt'][:-5], 1 - thry_data['c'][:-5], linewidth=0.5, c='black')
    ax3.plot(numerics_data['svals'][inds] * numerics_data['alpha_cxt'][n_idx, inds],
             1 - numerics_data['cvals'][n_idx, inds], '.', markersize=2, c='black')
    ax3.axhline(y=0.5, c='black', linestyle=':', linewidth=0.5)

    # panel (d): memory stabilities
    ax2.plot(thry_data['svals'][:-5] * thry_data['alpha_cxt'][:-5], thry_data['kappa_inacc'][:-5], linewidth=0.5,
             c='black')
    ax2.plot(thry_data['svals'][:-5] * thry_data['alpha_cxt'][:-5], thry_data['kappa_acc'][:-5], linewidth=0.5,
             c='black', linestyle='--')
    ax2.plot(numerics_data['svals'][inds] * numerics_data['alpha_cxt'][n_idx, inds],
             numerics_data['kappa_inacc'][n_idx, inds], '.', c='black', markersize=2)
    ax2.plot(numerics_data['svals'][inds] * numerics_data['alpha_cxt'][n_idx, inds],
             numerics_data['kappa_acc'][n_idx, inds], 'o', c='black', markersize=2,
             markerfacecolor='none', markeredgewidth=0.5)
    ax2.axhline(y=np.sqrt(2. / np.pi ** 2 / alpha_Delta(np.pi ** 2 / 2)), c='black', linestyle=':', linewidth=0.5)

    # panel (a): overlap
    ax4.plot(thry_data['svals'][:-5] * thry_data['alpha_cxt'][:-5],
             thry_data['m'][:-5], linewidth=0.5, c='black')
    ax4.axhline(y=m_Delta(np.pi ** 2 / 2), c='black', linestyle=':', linewidth=0.5)
    ax4.plot([16, 50], [m_Delta(np.pi ** 2 / 2), m_Delta(np.pi ** 2 / 2)], c='black', linewidth=0.5)

    # plot numerics for random weights at the right of the plot
    rand_x = 32
    ax1.plot([rand_x], [randgauss_data['alpha_cxt']][0][0], '.', markersize=2, color='black')
    ax3.plot([rand_x], [1-randgauss_data['cvals']][0][0], '.', markersize=2, color='black')
    ax2.plot([rand_x], [randgauss_data['kappa_acc']][0][0], 'o', markersize=2, color='black', markerfacecolor='none',
             markeredgewidth=0.5)
    ax2.plot([rand_x], [randgauss_data['kappa_inacc']][0][0], '.', markersize=2, color='black')

    # cut the x-axis to indicate this is the limit of infinite patterns
    p1 = Rectangle((12, -0.01), 10, 0.14, facecolor='white', edgecolor='none', zorder=50, clip_on=False)
    ax1.add_patch(p1)
    p2 = Rectangle((12, -0.2), 10, 3, facecolor='white', edgecolor='none', zorder=50, clip_on=False)
    ax2.add_patch(p2)
    p3 = Rectangle((12, -0.1), 10, 1, facecolor='white', edgecolor='none', zorder=50, clip_on=False)
    ax3.add_patch(p3)
    p4 = Rectangle((12, 0.45), 10, 0.5, facecolor='white', edgecolor='none', zorder=50, clip_on=False)
    ax4.add_patch(p4)

    # formatting
    ax3.legend(('Theory', 'Sim'), fontsize=pu.fs1, frameon=False, handlelength=1,
               loc='upper right', bbox_to_anchor=(0.7, 0.9))
    ax2.legend(('', '', r'$\bar{\kappa}_\text{acc}$', r'$\bar{\kappa}_\text{inacc}$'), fontsize=pu.fs1,
               frameon=False, handlelength=1, ncol=2, columnspacing=0.5,
               loc='upper right', bbox_to_anchor=(1.1, 0.5)).set_zorder(100)
    ax2.text(8, 1.4, 'Sim', fontsize=pu.fs1, zorder=100)  # 0.13
    ax2.text(2, 1.4, 'Theory', fontsize=pu.fs1, zorder=100)

    for ax in [ax1, ax2, ax3, ax4]:
        ax.set_xscale('log', base=2)
        ax.set_xticks([0.25, 0.5, 1, 2, 4, 8, 16, 32])
        ax.tick_params(axis='both', width=0.5, length=3, pad=1)
        ax.set_xticklabels(
            [r'${}^1{\mskip -5mu/\mskip -3mu}_4$', r'${}^1{\mskip -5mu/\mskip -3mu}_2$',
             '1', '2', '4', '8', ' ', 'r'], fontsize=pu.fs1)
        ax.set_xlim([0.125, 35])
    ax1.set_xlabel(r'Total load, $\alpha$', fontsize=pu.fs2)
    ax2.set_xlabel(r'Total load, $\alpha$', fontsize=pu.fs2)

    ax1.set_yticks([0, 0.07, 0.14])
    ax1.set_yticklabels(['0.00', '0.07', '0.14'], fontsize=pu.fs1)
    ax1.set_ylim([0, 0.14])
    ax1.set_ylabel('Single-cxt cap., 'r'$\alpha^*_\text{cxt}$', fontsize=pu.fs2)
    ax3.set_yticks([0, 0.5, 1.0])
    ax3.set_yticklabels(['0.0', '0.5', '1.0'], fontsize=pu.fs1)
    ax3.set_ylim([0, 1])
    ax3.set_ylabel('Syn. gating, 'r'$1$$-$$c$', fontsize=pu.fs2)
    ax2.set_yticks([0, 1, 2, 3])
    ax2.set_yticklabels(['0.0', '1.0', '2.0', '3.0'], fontsize=pu.fs1)
    ax2.set_ylim([0, 3])
    ax2.set_ylabel('Mean stability, 'r'$\bar{\kappa}$', fontsize=pu.fs2)
    ax4.set_yticks([0.5, 0.75, 1.0])
    ax4.set_yticklabels(['0.5', '0.75', '1.0'], fontsize=pu.fs1)
    ax4.set_ylim([0.5, 1.0])
    ax4.set_ylabel('Overlap, 'r'$m^*$', fontsize=pu.fs2)

    f.subplots_adjust(wspace=1, hspace=0.)
    sns.despine()
    f.tight_layout()

    ax1.text(-0.35, 1.1, '(c)', c='black', fontsize=8, transform=ax1.transAxes)
    ax2.text(-0.375, 1.1, '(d)', c='black', fontsize=8, transform=ax2.transAxes)
    ax3.text(-0.35, 1.1, '(b)', c='black', fontsize=8, transform=ax3.transAxes)
    ax4.text(-0.375, 1.1, '(a)', c='black', fontsize=8, transform=ax4.transAxes)

    if save_pdf:
        f.savefig(f"{pu.fig_path}/prx_fig_10.pdf", bbox_inches='tight', pad_inches=0)
    if show_plot:
        plt.show()


if __name__ == '__main__':
    main(save_pdf=True, show_plot=True)
