import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plot_utils as pu
import pickle
from scipy.stats import binom


# overlap information, f_m(Delta) [Eq. 11 in the paper]
def missinfo(m):
    return 0.5 * (1 + m) * np.log2(1 + m) + 0.5 * (1 - m) * np.log2(1 - m)


def main(save_pdf=False, show_plot=True):

    f, axs = plt.subplots(ncols=2, nrows=2, figsize=(8 * pu.cm, 7 * pu.cm), dpi=150)
    ax3 = axs[1, 0]
    ax4 = axs[1, 1]
    ax1 = axs[0, 0]
    ax2 = axs[0, 1]

    # load theory data
    with open('../data/theory/random_neuronal_gating_theory.pkl', 'rb') as tmp_file:
        nrn_theory_data = pickle.load(tmp_file)
    with open('../data/theory/random_synaptic_gating_theory.pkl', 'rb') as tmp_file:
        syn_theory_data = pickle.load(tmp_file)
    svals = nrn_theory_data['svals']
    avals = nrn_theory_data['avals']

    ###########################################################################
    # (1) OPTIMAL GATING PARAMETERS (panels a,b)
    ###########################################################################

    # panel (a): optimal gating for random neurons gating
    ax1.plot(svals[:], 1 - avals[np.argmax(nrn_theory_data['alpha'], axis=0)][:], '-', c=pu.nrn_clr,
             markersize=2, linewidth=0.75, alpha=0.75)

    ax1.plot(svals[:], 1 - avals[np.argmax(np.tile(avals, (100, 1)) * missinfo(
        nrn_theory_data['m'])*nrn_theory_data['alpha'], axis=0)][:],
            '-', c=pu.nrn_clr, markersize=2, linewidth=0.75, alpha=0.25)

    svals2 = np.insert(svals.astype(float), 1, 1.25)
    svals2[0] = 1.05
    ax1.plot(svals2, 1 - (np.sqrt(2. * svals2 - 1) - 1) / (svals2 - 1), c='black', linewidth=1,
             linestyle=':', alpha=0.75)

    # panel (b): optimal gating for random synaptic gating
    ax2.plot(svals[:], 1 - avals[np.argmax(syn_theory_data['alpha'], axis=0)][:],
             '-', c=pu.syn_clr, markersize=2, linewidth=0.75, alpha=0.75)

    ax2.plot(svals[:], 1 - avals[np.argmax(np.tile(avals, (100, 1)) * missinfo(
                                syn_theory_data['m']) * syn_theory_data['alpha'], axis=0)][:],
             '-', c=pu.syn_clr, markersize=2, linewidth=0.75, alpha=0.25)
    ax2.plot(svals, 1 - 1. / np.sqrt(svals), c='black', linewidth=1, linestyle=':', alpha=0.75)

    # formatting
    for ax in [ax1, ax2]:
        ax.set_ylim([0, 1.])
        ax.set_xlim([0, 99])
        ax.set_xticks([0, 49, 99])
        ax.set_xticklabels(['1', '50', '100'], fontsize=pu.fs1)
        ax.set_yticks([0, 0.5, 1.0])
        ax.set_yticklabels(['0.0', '0.5', '1.0'], fontsize=pu.fs1)
        ax.set_xlabel(r'Nr. contexts, $s$', fontsize=pu.fs2)
        ax.tick_params(axis='both', width=0.5, length=3, pad=1)
    ax1.set_ylabel('Optimal neuronal\n gating, ' + r'$1$$-$$a_{\text{opt}}$', fontsize=pu.fs2)
    ax2.set_ylabel('Optimal synaptic\n gating, ' + r'$1$$-$$c_{\text{opt}}$', fontsize=pu.fs2)

    ###########################################################################
    # (2) OVERLAP PER SYNAPSE (panels c,d)
    ###########################################################################
    avals = nrn_theory_data['avals']
    cvals = syn_theory_data['cvals']
    svals = np.array([1, 2, 10, 100])
    a_opt_vals = avals[np.argmax(nrn_theory_data['alpha'], axis=0)]
    c_opt_vals = avals[np.argmax(syn_theory_data['alpha'], axis=0)]
    a_opt_info_vals = avals[np.argmax(np.tile(avals, (100, 1)) * missinfo(
                                nrn_theory_data['m'])*nrn_theory_data['alpha'], axis=0)][:]
    c_opt_info_vals = avals[np.argmax(np.tile(avals, (100, 1)) * missinfo(
                                syn_theory_data['m'])*syn_theory_data['alpha'], axis=0)][:]
    avals2 = np.insert(nrn_theory_data['avals'], 0, 0)
    cvals2 = np.insert(syn_theory_data['cvals'], 0, 0)
    for i, s in enumerate(svals):

        ax3.semilogy(1-avals2, s * avals2 * avals2, c=pu.nrn_clrs5[i], linewidth=0.75, label=f'$s$={s}')
        ax4.semilogy(1-cvals2, s * cvals2, c=pu.syn_clrs5[i], linewidth=0.75, label=f'$s$={s}')

        a_opt = a_opt_vals[s - 1]
        ax3.plot([1. - a_opt], [s * a_opt * a_opt], 's', c=pu.nrn_clrs5[i], markersize=2)
        c_opt = c_opt_vals[s - 1]
        ax4.plot([1. - c_opt], [s * c_opt], 's', c=pu.syn_clrs5[i], markersize=2)

        a_opt = a_opt_info_vals[s - 1]
        ax3.plot([1. - a_opt], [s * a_opt * a_opt], 's', c=pu.nrn_clrs5[i], markerfacecolor='none', markersize=3,
                 markeredgewidth=0.5)
        c_opt = c_opt_info_vals[s - 1]
        ax4.plot([1. - c_opt], [s * c_opt], 's', c=pu.syn_clrs5[i], markerfacecolor='none', markersize=3,
                 markeredgewidth=0.5)

    ax3.plot(1 - avals, 1 + (1 - avals) ** 2, ':', c='gray', linewidth=1, alpha=0.75, label=r'$1$$+$$(1$$-$$a)^2$')
    ax4.plot(1 - cvals, 1. / cvals, ':', c='gray', linewidth=1, alpha=0.75, label=r'$1/c$')

    # formatting
    for ax in [ax3, ax4]:
        ax.set_xticks([0, 0.5, 1.0])
        ax.set_xticklabels(['0.0', '0.5', '1.0'], fontsize=pu.fs1)
        ax.set_yticks([0.01, 0.1, 1, 10, 100])
        ax.set_yticklabels([r'10$^{-2}$', r'10$^{-1}$', r'10$^0$', r'10$^1$', r'10$^2$'], fontsize=pu.fs1)
        ax.tick_params(axis='both', width=0.5, length=3, pad=1)
        ax.set_ylim([0.1, 100])
        ax.set_xlim([0, 1.0])
    ax3.set_ylabel('Shared contexts\n per synapse, 'r'$sa^2$', fontsize=pu.fs2)
    ax3.set_xlabel('Neuronal gating, 'r'$1$$-$$a$', fontsize=pu.fs2)
    ax4.set_ylabel('Shared contexts\n per synapse, 'r'$sc$', fontsize=pu.fs2)
    ax4.set_xlabel('Synaptic gating, 'r'$1$$-$$c$', fontsize=pu.fs2)

    f.subplots_adjust(wspace=1, hspace=0.)
    sns.despine()
    f.tight_layout()

    ###########################################################################
    # (3) BAR PLOT INSET FOR PANEL (c)
    ###########################################################################
    inset = ax3.inset_axes([0.65, 0.9, 0.4, 0.3])
    s = 10
    a = 0.4
    x = np.arange(s+1)
    y = binom.pmf(x, s, a**2)
    inset.bar(x, y, color=pu.nrn_clrs5[2], alpha=0.5)
    inset.axvline(x=s*a*a, c=pu.nrn_clrs5[2], linewidth=0.75, linestyle='--')
    inset.set_yticks([0, 0.2, 0.4])
    inset.set_yticklabels(['0.0', '0.2', '0.4'], fontsize=5)
    inset.set_xticks([0, 5, 10])
    inset.set_xticklabels(['0', '5', '10'], fontsize=5)
    inset.tick_params(axis='both', width=0.5, length=3, pad=1)
    inset.set_xlim([-0.65, 6])
    ax3.arrow(0.67, 0.47, 0.125, 0.2, color=pu.nrn_clrs5[2], linewidth=0.5, width=0.0001, head_width=0.025,
              head_length=0.025, transform=ax3.transAxes)

    ax1.text(-0.55, 1.1, '(a)', c='black', fontsize=8, transform=ax1.transAxes)
    ax2.text(-0.55, 1.1, '(b)', c='black', fontsize=8, transform=ax2.transAxes)
    ax3.text(-0.55, 1.1, '(c)', c='black', fontsize=8, transform=ax3.transAxes)
    ax4.text(-0.55, 1.1, '(d)', c='black', fontsize=8, transform=ax4.transAxes)

    ax1.legend((r'$a_{\text{opt }\alpha^*}$', r'$a_{\text{opt }I^*}$',
                r'$1$$-$$\frac{\sqrt{2s-1}-1}{s-1}$'), handlelength=1.5,
               fontsize=pu.fs1, frameon=False, loc='upper right', bbox_to_anchor=(0.9, 0.6))
    ax2.legend((r'$c_{\text{opt }\alpha^*}$', r'$c_{\text{opt }I^*}$',
                r'$1$$-$$s^{-1/2}$'), handlelength=1.5,  # r'$1$$-$$\sqrt{1/s}$'
               fontsize=pu.fs1, frameon=False, loc='upper right', bbox_to_anchor=(0.9, 0.5))

    ###########################################################################
    # (4) EXTENDED LEGENDS BELOW PANELS (c,d)
    ###########################################################################
    ax3.legend(ncol=1, frameon=False, handlelength=1.2, fontsize=pu.fs1, loc='upper right',
               bbox_to_anchor=(0.95, -0.5), markerfirst=False, columnspacing=0.5, handletextpad=1)
    ax4.legend(ncol=1, frameon=False, handlelength=1.2, fontsize=pu.fs1, loc='upper right',
               bbox_to_anchor=(0.95, -0.5), markerfirst=False, columnspacing=0.5, handletextpad=1)

    # extra text
    xoff = -0.1
    xoff2 = 0.275
    yoff = -0.61
    for i in range(4):
        ax3.plot([xoff2], [yoff - i * 0.125], c=pu.nrn_clrs4[i], marker='s', linestyle='None', markersize=3,
                 alpha=0.75, clip_on=False, markerfacecolor='none', markeredgewidth=0.5, transform=ax3.transAxes)
        ax4.plot([xoff2], [yoff - i * 0.125], c=pu.syn_clrs4[i], marker='s', linestyle='None', markersize=3,
                 alpha=0.75, clip_on=False, markerfacecolor='none', markeredgewidth=0.5, transform=ax4.transAxes)
        ax3.plot([xoff], [yoff - i * 0.125], c=pu.nrn_clrs4[i], marker='s', linestyle='None', markersize=2, alpha=0.75,
                 clip_on=False, transform=ax3.transAxes)
        ax4.plot([xoff], [yoff - i * 0.125], c=pu.syn_clrs4[i], marker='s', linestyle='None', markersize=2, alpha=0.75,
                 clip_on=False, transform=ax4.transAxes)

    ax3.text(-0.25, -0.515, r'$sa_{\text{opt }\alpha^*}^2$', fontsize=pu.fs1, color='black',
             clip_on=False, transform=ax3.transAxes)
    ax3.text(0.165, -0.515, r'$sa_{\text{opt }I^*}^2$', fontsize=pu.fs1, color='black',
             clip_on=False, transform=ax3.transAxes)
    ax4.text(-0.25, -0.515, r'$sc_{\text{opt }\alpha^*}$', fontsize=pu.fs1, color='black',
             clip_on=False, transform=ax4.transAxes)
    ax4.text(0.165, -0.515, r'$sc_{\text{opt }I^*}$', fontsize=pu.fs1, color='black',
             clip_on=False, transform=ax4.transAxes)
    ax3.text(0.75, -0.515, r'$sa^2$', fontsize=pu.fs1, color='black', clip_on=False, transform=ax3.transAxes)
    ax4.text(0.75, -0.515, r'$sc$', fontsize=pu.fs1, color='black', clip_on=False, transform=ax4.transAxes)

    for ax in [ax1, ax2]:
        box = ax.get_position()
        box.y0 = box.y0 + 0.01
        box.y1 = box.y1 - 0.0
        ax.set_position(box)

    for ax in [ax3, ax4]:
        box = ax.get_position()
        box.y0 = box.y0 - 0.0
        box.y1 = box.y1 - 0.03
        ax.set_position(box)

    if save_pdf:
        f.savefig(f"{pu.fig_path}/prx_fig_05.pdf", bbox_inches='tight', pad_inches=0)
    if show_plot:
        plt.show()


if __name__ == '__main__':
    main(save_pdf=True, show_plot=True)
