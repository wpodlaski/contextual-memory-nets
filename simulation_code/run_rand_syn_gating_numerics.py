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
a = 1.0
syn_ref = False
n_trials = 10
rand_cross = False
noise = 0.0
overlap_path = '../data/numerics/overlaps/random_synaptic_gating/'
linreg = LinearRegression()


def run_batch_random_synaptic_recall(cvals, svals, alpha_levels):

    for i, c in tqdm(enumerate(cvals)):
        for j, s in tqdm(enumerate(svals), leave=False):

            theory_data = tf.get_random_gating_theory(s, a, c)

            for ii, a0 in enumerate(alpha_levels):

                p = int(np.round(Ncxt * (a0 + theory_data['alpha_cxt'])))
                real_a0 = 1.0 * p / Ncxt - theory_data['alpha_cxt']
                net_prms = {'Ncxt': Ncxt, 'a': a, 'c': c, 's': s, 'syn_ref': syn_ref, 'rand_cross': rand_cross}
                if p < 0:
                    m, m2 = np.nan, np.nan
                else:
                    m, m2 = sf.simulate_recall(net_prms, p, noise, n_trials)

                sim_data = {'Ncxt': Ncxt,
                            'a': a,
                            'c': c,
                            's': s,
                            'syn_ref': syn_ref,
                            'n_trials': n_trials,
                            'a0_mult': a0,
                            'real_a0_mult': real_a0,
                            'sim_noise': noise,
                            'm': m,
                            'm2': m2}

                with open(overlap_path + f'Ncxt{Ncxt}/Ncxt{Ncxt}_c{int(100 * c)}_s{int(s)}'
                                         f'_ntrl{n_trials}_alevel{int(1000 * a0)}_nlevel{int(1000 * noise)}.pkl',
                          'wb') as tmp_file:
                    # noinspection PyTypeChecker
                    pickle.dump(sim_data, tmp_file)


def estimate_capacity_random_synaptic_gating(cvals, svals, alpha_levels, visualize_results=False):

    alpha_cxt = np.zeros((len(cvals), len(svals)))
    alpha = np.zeros((len(cvals), len(svals)))
    m_avg = np.zeros((len(cvals), len(svals), len(alpha_levels), ))
    alpha_mult = np.zeros((len(cvals), len(svals)))
    m_list = []

    for i, c in tqdm(enumerate(cvals)):
        for j, s in tqdm(enumerate(svals)):

            if visualize_results:
                f, axs = plt.subplots(nrows=2, ncols=4, figsize=(14, 4))
                axs = axs.flatten()
            else:
                f, axs = None, None

            theory_data = tf.get_random_gating_theory(s, a, c)
            m_theory = theory_data['m']

            real_alpha_levels = np.zeros_like(alpha_levels)
            for ii, a0 in enumerate(alpha_levels):

                # load data
                with open(f'data/numerics/overlaps/random_synaptic_gating/Ncxt{Ncxt}/Ncxt{Ncxt}_c{int(100 * c)}'
                          f'_s{int(s)}_ntrl{n_trials}_alevel{int(1000 * a0)}_nlevel{int(1000 * noise)}.pkl',
                          'rb') as file:
                    sim_data = pickle.load(file)

                m = np.array(sim_data['m'])
                m_avg[i, j, ii] = np.mean(m)
                m_list.append(m)
                real_alpha_levels[ii] = sim_data['real_a0_mult']

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

    capacity_data = {'cvals': cvals,
                     'svals': svals,
                     'n_trials': n_trials,
                     'alpha_levels': alpha_levels,
                     'alpha_cxt': alpha_cxt,
                     'alpha': alpha,
                     'm_avg': m_avg}

    with open(f'data/numerics/capacity/Ncxt{Ncxt}_rand_syn_gating_capacity.pkl', 'wb') as file:
        # noinspection PyTypeChecker
        pickle.dump(capacity_data, file)


def estimate_stability_random_synaptic_gating(cvals, svals):

    kappa_acc = np.zeros((len(cvals), len(svals)))
    kappa_var_acc = np.zeros((len(cvals), len(svals)))
    kappa_inacc = np.zeros((len(cvals), len(svals)))
    kappa_var_inacc = np.zeros((len(cvals), len(svals)))
    kappa_inacc0 = np.zeros((len(cvals), len(svals)))
    kappa_var_inacc0 = np.zeros((len(cvals), len(svals)))
    net_prms = {}

    for i, c in tqdm(enumerate(cvals)):
        for j, s in tqdm(enumerate(svals)):

            net_prms = {'N': Ncxt, 'a': a, 'c': c, 's': s, 'syn_ref': syn_ref}

            theory_data = tf.get_random_gating_theory(s, a, c)
            p = int(np.round(a * net_prms['N'] * theory_data['alpha_cxt']))
            (kappa_acc[i, j], kappa_var_acc[i, j],
                kappa_inacc[i, j], kappa_var_inacc[i, j],
                kappa_inacc0[i, j], kappa_var_inacc0[i, j]) = sf.simulate_stability(net_prms, p, n_trials)

    stability_data = {'cvals': cvals,
                      'svals': svals,
                      'n_trials': n_trials,
                      'kappa_acc': kappa_acc,
                      'kappa_var_acc': kappa_var_acc,
                      'kappa_inacc': kappa_inacc,
                      'kappa_var_inacc': kappa_var_inacc,
                      'kappa_inacc0': kappa_inacc0,
                      'kappa_var_inacc0': kappa_var_inacc0}

    with open(f"data/numerics/stability/N{net_prms['N']}_rand_syn_gating_stability.pkl", "wb") as file:
        # noinspection PyTypeChecker
        pickle.dump(stability_data, file)


def main():

    cvals = np.array([0.05, 0.15, 0.25, 0.35, 0.45, 0.65, 0.85])
    svals = np.array([1, 2, 5, 10, 50, 100])
    alpha_levels = np.array([-0.02, -0.01, -0.005, -0.001, 0.0, 0.001, 0.005, 0.01])

    run_batch_random_synaptic_recall(cvals, svals, alpha_levels)
    estimate_capacity_random_synaptic_gating(cvals, svals, alpha_levels)
    estimate_stability_random_synaptic_gating(cvals, svals)


if __name__ == '__main__':
    main()
