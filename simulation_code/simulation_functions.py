
import torch
import numpy as np
from cxt_mod_hopfield import CxtModNet


def simulate_recall(net_prms, p, noise_level, n_trials):

    Ncxt = net_prms['Ncxt']
    a = net_prms['a']
    c = net_prms['c']
    s = net_prms['s']
    syn_ref = net_prms['syn_ref']
    rand_cross = net_prms['rand_cross']
    T = 50  # net_prms['T']
    sync = True  # net_prms['sync']

    if 'randgauss' in net_prms and net_prms['randgauss']:
        noise_type = 'randgauss'
    else:
        noise_type = 'real'

    m = []
    m2 = []
    syn_ref_c = []
    for n in range(n_trials):
        net = CxtModNet(Ncxt=Ncxt, a=a, c=c, s=s, syn_ref=syn_ref, device='cuda', rand_cross=rand_cross)
        c_tmp = net.learn_patterns(p, noise_type=noise_type, noise_level=0.)
        m_tmp, m2_tmp = net.test_memory_recall(T, mean=False, noise_level=noise_level, sync=sync)  # previously T=200
        m += list(m_tmp)
        m2 += list(m2_tmp)
        syn_ref_c.append(c_tmp)
        net.clear_memory()

    if syn_ref:
        return m, m2, np.mean(syn_ref_c)
    else:
        return m, m2


def simulate_stability(net_prms, p, n_trials):

    N = net_prms['N']
    a = net_prms['a']
    c = net_prms['c']
    s = net_prms['s']
    syn_ref = net_prms['syn_ref']

    if 'randgauss' in net_prms and net_prms['randgauss']:
        noise_type = 'randgauss'
    else:
        noise_type = 'real'

    k_acc, var_k_acc = np.zeros((n_trials,)), np.zeros((n_trials,))
    k_inacc, var_k_inacc = np.zeros((n_trials,)), np.zeros((n_trials,))
    k_inacc0, var_k_inacc0 = np.zeros((n_trials,)), np.zeros((n_trials,))
    for n in range(n_trials):
        net = CxtModNet(N=N, a=a, c=c, s=s, syn_ref=syn_ref, device='cuda')
        _ = net.learn_patterns(p, noise_type=noise_type, noise_level=0.)
        k_acc[n], var_k_acc[n] = net.get_accessible_stability()
        if s > 1:
            k_inacc[n], var_k_inacc[n], k_inacc0[n], var_k_inacc0[n] = net.get_inaccessible_stability(p)
        else:
            k_inacc[n], var_k_inacc[n], k_inacc0[n], var_k_inacc0[n] = np.nan, np.nan, np.nan, np.nan

        net.clear_memory()

    return (np.mean(k_acc), np.mean(var_k_acc), np.mean(k_inacc), np.mean(var_k_inacc),
            np.mean(k_inacc0), np.mean(var_k_inacc0))
