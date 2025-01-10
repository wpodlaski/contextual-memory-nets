
import numpy as np
import pickle
import theory_functions as tf


def run_random_neuronal_gating_theory():

    print('Running random neuronal gating theory......', end="")

    avals = np.linspace(0.01, 1.0, 100)
    svals = np.arange(1, 101)
    c = 1.0

    alpha_cxt = np.zeros((avals.shape[0], svals.shape[0]))
    alpha = np.zeros((avals.shape[0], svals.shape[0]))
    kappa_acc = np.zeros((avals.shape[0], svals.shape[0]))
    kappa_inacc = np.zeros((avals.shape[0], svals.shape[0]))
    m = np.zeros((avals.shape[0], svals.shape[0]))

    for i, a in enumerate(avals):
        for j, s in enumerate(svals):

            thry_data = tf.get_random_gating_theory(s, a, c)
            alpha_cxt[i, j] = thry_data['alpha_cxt']
            alpha[i, j] = thry_data['alpha']
            kappa_acc[i, j] = thry_data['kappa_acc']
            kappa_inacc[i, j] = thry_data['kappa_inacc']
            m[i, j] = thry_data['m']

    theory_data = {'avals': avals,
                   'svals': svals,
                   'alpha_cxt': alpha_cxt,
                   'alpha': alpha,
                   'm': m,
                   'kappa_acc': kappa_acc,
                   'kappa_inacc': kappa_inacc}

    with open(f'data/theory/random_neuronal_gating_theory.pkl', 'wb') as tmp_file:
        # noinspection PyTypeChecker
        pickle.dump(theory_data, tmp_file)

    print('Done')

    return theory_data


def run_random_synaptic_gating_theory():

    print('Running random synaptic gating theory......', end="")

    cvals = np.linspace(0.01, 1.0, 100)
    svals = np.arange(1, 101)
    a = 1.0

    alpha_cxt = np.zeros((cvals.shape[0], svals.shape[0]))
    alpha = np.zeros((cvals.shape[0], svals.shape[0]))
    kappa_acc = np.zeros((cvals.shape[0], svals.shape[0]))
    kappa_inacc = np.zeros((cvals.shape[0], svals.shape[0]))
    m = np.zeros((cvals.shape[0], svals.shape[0]))

    for i, c in enumerate(cvals):
        for j, s in enumerate(svals):

            thry_data = tf.get_random_gating_theory(s, a, c)
            alpha_cxt[i, j] = thry_data['alpha_cxt']
            alpha[i, j] = thry_data['alpha']
            kappa_acc[i, j] = thry_data['kappa_acc']
            kappa_inacc[i, j] = thry_data['kappa_inacc']
            m[i, j] = thry_data['m']

    theory_data = {'cvals': cvals,
                   'svals': svals,
                   'alpha_cxt': alpha_cxt,
                   'alpha': alpha,
                   'kappa_acc': kappa_acc,
                   'kappa_inacc': kappa_inacc,
                   'm': m}

    with open(f'data/theory/random_synaptic_gating_theory.pkl', 'wb') as tmp_file:
        # noinspection PyTypeChecker
        pickle.dump(theory_data, tmp_file)

    print('Done')

    return theory_data


def run_synaptic_refinement_random_neuronal_theory():

    print('Running synaptic refinement theory......', end="")

    avals = np.linspace(0.01, 1.0, 100)
    svals = np.arange(1, 101)

    alpha_cxt = np.zeros((avals.shape[0], svals.shape[0]))
    alpha = np.zeros((avals.shape[0], svals.shape[0]))
    m = np.zeros((avals.shape[0], svals.shape[0]))
    c = np.zeros((avals.shape[0], svals.shape[0]))
    kappa_acc = np.zeros((avals.shape[0], svals.shape[0]))
    kappa_inacc = np.zeros((avals.shape[0], svals.shape[0]))

    for i, a in enumerate(avals):
        for j, s in enumerate(svals):

            thry_data = tf.get_synaptic_refinement_theory(s=s, a=a)
            alpha_cxt[i, j] = thry_data['alpha_cxt']
            alpha[i, j] = thry_data['alpha']
            m[i, j] = thry_data['m']
            c[i, j] = thry_data['c']
            kappa_acc[i, j] = thry_data['kappa_acc']
            kappa_inacc[i, j] = thry_data['kappa_inacc']

    theory_data = {'avals': avals,
                   'svals': svals,
                   'alpha_cxt': alpha_cxt,
                   'alpha': alpha,
                   'm': m,
                   'c': c,
                   'kappa_acc': kappa_acc,
                   'kappa_inacc': kappa_inacc,
                   }

    with open('data/theory/synaptic_refinement_random_neuronal_theory.pkl', 'wb') as tmp_file:
        # noinspection PyTypeChecker
        pickle.dump(theory_data, tmp_file)

    print('Done')

    return theory_data


def run_synaptic_refinement_arbitrary_theory():

    print('Running arbitrary synaptic refinement theory......', end="")

    a = 1
    svals = np.logspace(0, 9, 100, base=2)

    alpha_cxt = np.zeros((svals.shape[0],))
    alpha = np.zeros((svals.shape[0],))
    kappa_acc = np.zeros((svals.shape[0]))
    kappa_inacc = np.zeros((svals.shape[0]))
    m = np.zeros((svals.shape[0],))
    c = np.zeros((svals.shape[0],))

    for j, s in enumerate(svals):

        thry_data = tf.get_synaptic_refinement_theory(s=s)
        alpha_cxt[j] = thry_data['alpha_cxt']
        alpha[j] = thry_data['alpha']
        kappa_acc[j] = thry_data['kappa_acc']
        kappa_inacc[j] = thry_data['kappa_inacc']
        m[j] = thry_data['m']
        c[j] = thry_data['c']

    theory_data = {'a': a,
                   'svals': svals,
                   'mean_alpha_cxt': alpha_cxt,
                   'mean_alpha': alpha,
                   'mean_kappa_acc': kappa_acc,
                   'mean_kappa_inacc': kappa_inacc,
                   'mean_m': m,
                   'mean_c': c}

    with open('data/theory/synaptic_refinement_arbitrary_theory.pkl', 'wb') as tmp_file:
        # noinspection PyTypeChecker
        pickle.dump(theory_data, tmp_file)

    print('Done')

    return theory_data


def run_all_theories():
    _ = run_random_neuronal_gating_theory()
    _ = run_random_synaptic_gating_theory()
    _ = run_synaptic_refinement_random_neuronal_theory()
    _ = run_synaptic_refinement_arbitrary_theory()


def main():
    run_all_theories()


if __name__ == '__main__':
    main()
