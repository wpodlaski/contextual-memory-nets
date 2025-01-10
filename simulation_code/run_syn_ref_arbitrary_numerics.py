import numpy as np
import theory_functions as tf
import simulation_functions as sf
import pickle
from tqdm import tqdm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns


# global parameters for all functions
Ncxt = 4000
syn_ref = False
n_trials = 10
rand_cross = False
noise = 0.0
overlap_path = '../data/numerics/overlaps/synaptic_refinement_arbitrary/'
linreg = LinearRegression()


def run_batch_synaptic_refinement_arbitrary_recall(avals, svals, alpha_levels):

    for i, a in tqdm(enumerate(avals)):
        for j, s in tqdm(enumerate(svals), leave=False):

            theory_data = tf.get_synaptic_refinement_theory(s, a)

            for ii, a0 in enumerate(alpha_levels):

                p = int(np.round(Ncxt * (a0 + theory_data['mean_alpha_cxt'])))
                real_a0 = 1.0 * p / Ncxt - theory_data['mean_alpha_cxt']

                net_prms = {'Ncxt': Ncxt, 'a': a, 'c': 1.0, 's': s, 'syn_ref': syn_ref, 'rand_cross': rand_cross}

                if p < 0:
                    m, m2, syn_ref_c = np.nan, np.nan, np.nan
                else:
                    m, m2, syn_ref_c = sf.simulate_recall(net_prms, p, noise, n_trials)

                sim_data = {'Ncxt': Ncxt,
                            'a': a,
                            'c': syn_ref_c,
                            's': s,
                            'syn_ref': syn_ref,
                            'n_trials': n_trials,
                            'a0_mult': a0,
                            'real_a0_mult': real_a0,
                            'sim_noise': noise,
                            'm': m,
                            'm2': m2}

                with open(f'data/numerics/overlaps/synaptic_refinement_arbitrary/Ncxt{Ncxt}/'
                          f'Ncxt{Ncxt}_s{int(s)}_ntrl{n_trials}_alevel{int(1000 * a0)}_'
                          f'nlevel{int(1000 * noise)}.pkl', 'wb') as tmp_file:
                    # noinspection PyTypeChecker
                    pickle.dump(sim_data, tmp_file)


def estimate_capacity_synaptic_refinement_arbitrary_gating(avals, svals, alpha_levels, visualize_results=False):

    cvals = np.zeros((len(avals), len(svals)))
    alpha_cxt = np.zeros((len(avals), len(svals)))
    alpha = np.zeros((len(avals), len(svals)))
    m_avg = np.zeros((len(avals), len(svals), len(alpha_levels), ))
    c_avg = np.zeros((len(avals), len(svals), len(alpha_levels),))
    alpha_mult = np.zeros((len(avals), len(svals)))
    m_list = []

    for i, a in tqdm(enumerate(avals)):
        for j, s in tqdm(enumerate(svals), leave=False):

            if visualize_results:
                f, axs = plt.subplots(nrows=2, ncols=4, figsize=(14, 5))
                axs = axs.flatten()
            else:
                f, axs = None, None

            theory_data = tf.get_synaptic_refinement_theory(s, a)
            m_theory = theory_data['mean_m']

            real_alpha_levels = np.zeros_like(alpha_levels)
            for ii, a0 in enumerate(alpha_levels):

                # load data
                with open(f'data/numerics/overlaps/synaptic_refinement_arbitrary/Ncxt{Ncxt}/'
                          f'Ncxt{Ncxt}_s{int(s)}_ntrl{n_trials}_alevel{int(1000 * a0)}_'
                          f'nlevel{int(1000 * noise)}.pkl', 'rb') as tmp_file:
                    sim_data = pickle.load(tmp_file)

                m = np.array(sim_data['m'])
                m_avg[i, j, ii] = np.mean(m)
                m_list.append(m)
                real_alpha_levels[ii] = sim_data['real_a0_mult']
                c = np.array(sim_data['c'])
                c_avg[i, j, ii] = np.mean(c)

                if visualize_results:
                    (y, x, _) = axs[ii].hist(m, bins=np.linspace(0, 1, 201), color='gray', alpha=0.75)
                    axs[ii].axvline(x=m_theory, c='black')
                    axs[ii].axvline(x=np.mean(m), c='black')
                    axs[ii].set_yscale('log')
                    axs[ii].set_title("a0=%.5f" % sim_data['real_a0_mult'])

            if visualize_results:
                f.suptitle(f'a={a}, s={s}')
                plt.tight_layout()
                sns.despine()
                plt.show()

            tmp_m_avg = m_avg[i, j, :]
            tmp_alpha_levels = real_alpha_levels.copy()

            # estimate the closest points above and below the theoretical overlap value
            y1 = tmp_m_avg - m_theory
            y1[y1 < 0] = np.nan
            x1 = tmp_alpha_levels[np.nanargmin(y1)]
            y1 = tmp_m_avg[np.nanargmin(y1)]
            y2 = m_theory - tmp_m_avg
            y2[y2 < 0] = np.nan
            x2 = tmp_alpha_levels[np.nanargmin(y2)]
            y2 = tmp_m_avg[np.nanargmin(y2)]

            # estimate the capacity at the theoretical overlap
            linreg.fit(np.array([y1, y2]).reshape(-1, 1), np.array([x1, x2]))
            mult = linreg.predict(np.array([[m_theory]]))[0]
            alpha_mult[i, j] = mult
            alpha_cxt[i, j] = mult + theory_data['alpha_cxt']
            alpha[i, j] = a * s * alpha_cxt[i, j]

    capacity_data = {'syn_ref': syn_ref,
                     'avals': avals,
                     'svals': svals,
                     'cvals': cvals,
                     'n_trials': n_trials,
                     'alpha_levels': alpha_levels,
                     'alpha_cxt': alpha_cxt,
                     'alpha': alpha,
                     'm_avg': m_avg}

    with open(f'data/numerics/capacity/Ncxt{Ncxt}_syn_ref_arbitrary_capacity.pkl', 'wb') as tmp_file:
        # noinspection PyTypeChecker
        pickle.dump(capacity_data, tmp_file)


def estimate_stability_synaptic_refinement_arbitrary_gating(avals, svals):

    kappa_acc = np.zeros((len(avals), len(svals)))
    kappa_var_acc = np.zeros((len(avals), len(svals)))
    kappa_inacc = np.zeros((len(avals), len(svals)))
    kappa_var_inacc = np.zeros((len(avals), len(svals)))
    kappa_inacc0 = np.zeros((len(avals), len(svals)))
    kappa_var_inacc0 = np.zeros((len(avals), len(svals)))
    net_prms = None

    for i, a in tqdm(enumerate(avals)):
        for j, s in tqdm(enumerate(svals)):

            net_prms = {'N': Ncxt, 'a': a, 'c': 1.0, 's': s, 'syn_ref': syn_ref}

            theory_data = tf.get_synaptic_refinement_theory(s=s, a=a)
            p = int(np.round(a * net_prms['N'] * theory_data['alpha_cxt']))
            (kappa_acc[i, j], kappa_var_acc[i, j],
             kappa_inacc[i, j], kappa_var_inacc[i, j],
             kappa_inacc0[i, j], kappa_var_inacc0[i, j]) = sf.simulate_stability(net_prms, p, n_trials)

    stability_data = {'avals': avals,
                      'svals': svals,
                      'n_trials': n_trials,
                      'kappa_acc': kappa_acc,
                      'kappa_var_acc': kappa_var_acc,
                      'kappa_inacc': kappa_inacc,
                      'kappa_var_inacc': kappa_var_inacc,
                      'kappa_inacc0': kappa_inacc0,
                      'kappa_var_inacc0': kappa_var_inacc0}

    with open(f"data/numerics/stability/N{net_prms['N']}_syn_ref_arbitrary_stability.pkl", "wb") as tmp_file:
        # noinspection PyTypeChecker
        pickle.dump(stability_data, tmp_file)


def main():
    avals = np.array([1.0])
    svals = np.array([1, 2, 4, 8, 16, 32, 64, 128])
    alpha_levels = np.array([-0.02, -0.01, -0.005, -0.001, 0.0, 0.001, 0.005, 0.01])

    run_batch_synaptic_refinement_arbitrary_recall(avals, svals, alpha_levels)
    estimate_capacity_synaptic_refinement_arbitrary_gating(avals, svals, alpha_levels)
    estimate_stability_synaptic_refinement_arbitrary_gating(avals, svals)


if __name__ == '__main__':
    main()
