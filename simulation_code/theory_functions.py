import numpy as np
import pickle


# load noisy Hopfield capacity and overlap
with open('../figure_generation_code/data/gauss_hopfield.pkl', 'rb') as file:
    gauss_data = pickle.load(file)
alpha_Delta = gauss_data['alph_Delta']
m_Delta = gauss_data['m_Delta']


def get_random_gating_theory(s, a, c):

    # see Eqs. 22-24 of the paper
    noise_numerator = 1 + (s - 1) * a * a * c
    noise_denominator = 1 + (s - 1) * a * a * a * a * c * c * c * c
    m = m_Delta(noise_numerator / noise_denominator / c)  # Eq. 23
    alpha_cxt = alpha_Delta(noise_numerator / noise_denominator / c) / noise_denominator  # Eq. 24

    alpha = alpha_cxt * a * s  # Eq. 4
    kappa_acc = np.sqrt(c / (noise_numerator * alpha_cxt))  # Eq. 25
    kappa_inacc = a * a * c * np.sqrt(c / (noise_numerator * alpha_cxt))  # Eq. 26

    return {'alpha_cxt': alpha_cxt,
            'alpha': alpha,
            'm': m,
            'kappa_acc': kappa_acc,
            'kappa_inacc': kappa_inacc}


def get_synaptic_refinement_theory(s=1, a=1.0):

    def c_fun(k_):  # Eq. 33
        return 1 - (1. / np.pi) * np.arctan(np.sqrt(k_))

    k = (s - 1) * a * a
    synref_c = c_fun(k)  # synaptic gating due to synaptic refinement
    noise = ((1 + k) * np.pi ** 2 * synref_c + np.pi * np.sqrt(k)) / (np.pi * synref_c + np.sqrt(k)) ** 2
    m = m_Delta(noise)  # Eq. 40
    alpha_cxt = alpha_Delta(noise)  # Eq. 41
    alpha = alpha_cxt * a * s  # Eq. 4
    kappa_acc = np.sqrt(1.0 / (noise * alpha_cxt))  # Eq. 43
    kappa_inacc = a * a * synref_c / np.sqrt(alpha_cxt * ((1 + k) * synref_c + np.sqrt(k) / np.pi))  # Eq. 44

    return {'alpha_cxt': alpha_cxt,
            'alpha': alpha,
            'm': m,
            'c': synref_c,
            'kappa_acc': kappa_acc,
            'kappa_inacc': kappa_inacc}
