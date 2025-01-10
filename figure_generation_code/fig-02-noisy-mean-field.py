import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plot_utils as pu
import pickle


# overlap information, f_m(Delta) [Eq. 11 in the paper]
def missinfo(m):
    return 0.5 * (1 + m) * np.log2(1 + m) + 0.5 * (1 - m) * np.log2(1 - m)


def main(save_pdf=False, show_plot=True):

    # set up figure panels
    f, axs = plt.subplots(ncols=2, nrows=2, figsize=(8 * pu.cm, 6 * pu.cm), dpi=150)
    ax2 = axs[0, 0]
    ax1 = axs[0, 1]
    ax3 = axs[1, 1]
    ax4 = axs[1, 0]

    # load theoretical values
    with open('../data/theory/gauss_hopfield.pkl', 'rb') as file:
        gauss_data = pickle.load(file)
    alpha_Delta = gauss_data['alph_Delta']
    m_Delta = gauss_data['m_Delta']
    Delta = np.linspace(0, 50, 401)

    # panel (a): capacity
    ax1.semilogx(1+Delta**2, m_Delta(1+Delta**2), c='black', linewidth=0.75)
    ax1.set_yticks([0, 0.5, 1.0])
    ax1.set_yticklabels(['0.0', '0.5', '1.0'], fontsize=pu.fs1)
    ax1.set_ylabel(r'Overlap, $m^*_\Delta$', fontsize=pu.fs2)

    # panel (b): overlap
    ax2.semilogx(1+Delta**2, alpha_Delta(1+Delta**2), c='black', linewidth=0.75)
    ax2.set_yticks([0, 0.07, 0.14])
    ax2.set_yticklabels(['0.0', '0.07', '0.14'], fontsize=pu.fs1)
    ax2.set_ylabel(r'Capacity, $\alpha^*_\Delta$', fontsize=pu.fs2)
    ax2.set_ylim([0, 0.145])

    # panel (d): overlap info
    ax3.semilogx(1 + Delta ** 2, missinfo(m_Delta(1 + Delta ** 2)), c='black', linewidth=0.75)
    ax3.set_yticks([0, 0.5, 1.0])
    ax3.set_yticklabels(['0.0', '0.5', '1.0'], fontsize=pu.fs1)
    ax3.set_ylabel(r'Overlap info., $f_m$', fontsize=pu.fs2)

    # panel (c): info content
    ax4.semilogx(1 + Delta ** 2, alpha_Delta(1+Delta**2) * missinfo(m_Delta(1 + Delta ** 2)), c='black', linewidth=0.75)
    ax4.set_yticks([0, 0.07, 0.14])
    ax4.set_yticklabels(['0.0', '0.07', '0.14'], fontsize=pu.fs1)
    ax4.set_ylabel(r'Info. content, $I^*$', fontsize=pu.fs2)
    ax4.set_ylim([0, 0.145])

    # figure formatting
    for ax in axs.flatten():
        ax.set_xticks([1e0, 1e1, 1e2, 1e3])
        ax.set_xticklabels([r'10$^0$', r'10$^1$', r'10$^2$', r'10$^3$'], fontsize=pu.fs1)
        ax.set_xlim([1, 2500])
        ax.tick_params(axis='both', width=0.5, length=3, pad=1)
    ax3.set_xlabel(r'Noise level, $1 + \Delta_0^2$', fontsize=pu.fs2)
    ax4.set_xlabel(r'Noise level, $1 + \Delta_0^2$', fontsize=pu.fs2)

    f.subplots_adjust(wspace=1, hspace=0.)
    sns.despine()
    f.tight_layout()

    ax1.text(-0.35, 1.1, '(b)', c='black', fontsize=8, transform=ax1.transAxes)
    ax2.text(-0.35, 1.1, '(a)', c='black', fontsize=8, transform=ax2.transAxes)
    ax3.text(-0.35, 1.1, '(d)', c='black', fontsize=8, transform=ax3.transAxes)
    ax4.text(-0.35, 1.1, '(c)', c='black', fontsize=8, transform=ax4.transAxes)

    if save_pdf:
        f.savefig(f"{pu.fig_path}/prx_fig_02.pdf", bbox_inches='tight', pad_inches=0)
    if show_plot:
        plt.show()


if __name__ == '__main__':
    main(save_pdf=True, show_plot=True)
