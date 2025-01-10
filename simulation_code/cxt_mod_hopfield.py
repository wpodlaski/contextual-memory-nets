
import torch
import numpy as np


class CxtModNet:

    def __init__(self, N=None, Ncxt=None, a=1.0, c=1.0, s=1, syn_ref=False,
                 rand_cross=False, device='cpu'):
        """
        :param N: Network size (NOTE: either N or Ncxt should be specified, not both)
        :param Ncxt: Subnetwork size, with neuronal gating
        :param a: Random neuronal gating level
        :param c: Random synaptic gating level
        :param s: Number of contexts
        :param syn_ref: Indicates if synaptic refinement should be applied
        :param rand_cross: (*FOR TESTING*) Indicates if crosstalk should be randomly permuted or not (default is not)
        :param device: Indicates 'cpu' or 'cuda'
        """

        # check for GPU
        if not torch.cuda.is_available():
            self.device = 'cpu'
            raise Warning('Warning: GPU not found, running on CPU.')
        else:
            self.device = device

        # network parameters
        self.a = a
        self.c = c
        self.s = s
        if N is None and Ncxt is None:
            raise Exception('Either N or Ncxt must be specified!')
        elif N is not None and Ncxt is not None:
            raise Exception('Only one of N or Ncxt can be specified, not both!')
        elif N is not None:
            self.N = N
            self.Ncxt = int(a*N)
        else:
            self.Ncxt = Ncxt
            self.N = int(1.0 * Ncxt / a)
        self.eta = 1. / (a * c * self.N)  # normalization: note, this is not so important for the numerics
        self.syn_ref = syn_ref
        self.randomize_crosstalk = rand_cross

        # pattern parameters -- for now not set, just initialized
        self.p, self.P, self.Xi, self.W = None, None, None, None
        self.syn_ref_c, self.stability = None, None

        # to optimize the code: separate functions depending on level of neuronal, synaptic gating & refinement
        if self.s == 1:
            self.generate_crosstalk = self.initialize_weights
            self.get_inaccessible_stability = lambda p: None
        else:
            self.generate_crosstalk = {(True, True): None,
                                       (True, False): self.generate_random_nrn_crosstalk,
                                       (False, True): self.generate_random_syn_crosstalk,
                                       (False, False): self.generate_standard_crosstalk}[(self.a < 1, self.c < 1)]
            self.get_inaccessible_stability = {(True, True, False): lambda p: None,
                                               (True, False, False): self.get_random_nrn_inaccessible_stability,
                                               (False, True, False): self.get_random_syn_inaccessible_stability,
                                               (False, False, False): self.get_random_nrn_inaccessible_stability,
                                               (True, True, True): lambda p: None,
                                               (True, False, True): self.get_random_nrn_inaccessible_stability,
                                               (False, True, True): lambda p: None,
                                               (False, False, True): self.get_random_nrn_inaccessible_stability
                                               }[(self.a < 1, self.c < 1, self.syn_ref)]

    ################################################################################################
    # LEARN PATTERNS: main function for setting the synaptic weights
    ################################################################################################
    def learn_patterns(self, p, noise_type='real', noise_level=0.):

        self.p = p
        self.P = int(self.s * self.p)

        # set weights with crosstalk plus accessible patterns
        # NOTE: accessible patterns have no neuronal gating because we are only simulating the accessible neurons
        self.generate_crosstalk()
        if self.randomize_crosstalk:
            idx = torch.randperm(self.W.nelement())
            self.W = self.W.view(-1)[idx].view(self.W.size())
        self.Xi = -1 + 2 * torch.bernoulli(0.5 * torch.ones((self.p, self.Ncxt), device=self.device))
        self.W += self.eta * torch.t(self.Xi) @ self.Xi

        # add random noise
        self.W += noise_level * torch.randn(size=(self.Ncxt, self.Ncxt), device=self.device)

        # remove self-connections
        self.W.as_strided([self.Ncxt], [self.Ncxt + 1]).copy_(torch.diag(self.W) * 0.0)

        # random synaptic gating
        if self.c < 1 and noise_type == 'real':
            syn_mask = torch.bernoulli(self.c * torch.ones((self.Ncxt, self.Ncxt),
                                                           device=self.device)).bool()
            self.W *= syn_mask
            del syn_mask
        # *TESTING ONLY*: add the equivalent amount of noise to the weights instead of dilution (Sompolinsky 1986)
        elif self.c < 1 and noise_type == 'ideal':
            self.W += (1. / self.c) * np.sqrt(self.p * (1 - self.c) / self.c / self.Ncxt / self.Ncxt) * torch.randn(
                                                                        size=(self.Ncxt, self.Ncxt), device=self.device)

        # random gaussian weights
        if noise_type == 'randgauss':
            self.W = torch.normal(mean=torch.zeros((self.Ncxt, self.Ncxt), device=self.device))
            self.W.as_strided([self.Ncxt], [self.Ncxt + 1]).copy_(torch.diag(self.W) * 0.0)

        # synaptic refinement
        if self.syn_ref:
            W0 = torch.t(self.Xi) @ self.Xi
            M = (W0 * self.W) + 1e-8 * torch.normal(mean=torch.zeros((self.Ncxt, self.Ncxt), device=self.device)) > 0.
            self.W *= M.float()
            self.syn_ref_c = torch.mean(M[torch.triu_indices(self.Ncxt, 1)].float()).item()
            del M

        # return the level of synaptic gating (only relevant for synaptic refinement)
        return self.syn_ref_c

    def initialize_weights(self):
        self.W = torch.zeros((self.Ncxt, self.Ncxt), device=self.device)

    ################################################################################################
    # CROSSTALK FUNCTIONS: add the patterns from other contexts
    ################################################################################################
    def generate_standard_crosstalk(self):
        SXi = -1 + 2 * torch.bernoulli(0.5 * torch.ones((self.P - self.p, self.Ncxt), device=self.device))
        self.W = self.eta * torch.t(SXi) @ SXi
        del SXi

    def generate_random_nrn_crosstalk(self):
        nrn_mask = torch.bernoulli(self.a * torch.ones((self.P - self.p, self.Ncxt), device=self.device)).bool()
        SXi = nrn_mask * (-1 + 2 * torch.bernoulli(0.5 * torch.ones((self.P - self.p, self.Ncxt), device=self.device)))
        self.W = self.eta * torch.t(SXi) @ SXi
        del nrn_mask, SXi

    def generate_random_syn_crosstalk(self):
        self.initialize_weights()
        for i in range(self.s - 1):
            SXi = -1 + 2 * torch.bernoulli(0.5 * torch.ones((self.p, self.Ncxt), device=self.device))
            Wtmp = self.eta * torch.t(SXi) @ SXi
            syn_mask = torch.bernoulli(self.c * torch.ones((self.Ncxt, self.Ncxt),
                                                           device=self.device)).bool()
            self.W += syn_mask * Wtmp
            del SXi, Wtmp, syn_mask

    ################################################################################################
    # RECALL: main function to test memory recall performance
    ################################################################################################
    def test_memory_recall(self, T, sync=True, mean=True, noise_level=0.):
        X = self.Xi.clone()

        # Default: synchronous dynamics
        if sync:
            for t in range(T):  # run for T time steps
                X = torch.sign(X @ self.W + 1e-8 * torch.randn_like(X))

                # *TESTING ONLY*: can be run with noisy dynamics
                # if noise_level > 0.:
                #     X *= (-1 + 2 * torch.bernoulli((1 - noise_level) * torch.ones_like(X, device=self.device)))

            # run one additional step to see if it is a fixed point or two-step
            X2 = torch.sign(X @ self.W + 1e-8 * torch.randn_like(X))

        # *TESTING ONLY*: asynchronous dynamics
        else:  # async
            for t in range(T):  # run for T time steps
                r = np.random.permutation(self.Ncxt)
                for n in np.random.permutation(self.Ncxt):
                    X[:, n] = torch.sign(X @ self.W[:, n] + 1e-8 * torch.randn_like(X[:, n]))  # , device=self.device)
            X2 = X.clone()
            for n in np.random.permutation(self.Ncxt):
                X2[:, n] = torch.sign(X2 @ self.W[:, n] + 1e-8 * torch.randn_like(X2[:, n]))  # , device=self.device)

        # either return the mean overlap or all individual overlap values
        if mean:
            return torch.mean(torch.sum(X * self.Xi, 1) / self.Ncxt).item()
        else:
            return (torch.sum(X * self.Xi, 1).numpy(force=True) / self.Ncxt,
                    torch.sum(X * X2, 1).numpy(force=True) / self.Ncxt)

    ################################################################################################
    # MEMORY STABILITY FUNCTIONS: test stability of accessible or inaccessible memories
    ################################################################################################
    def get_accessible_stability(self):
        self.stability = torch.div(self.Xi * (self.Xi @ self.W), torch.norm(self.W, dim=0))
        return torch.mean(self.stability).item(), torch.var(self.stability).item()

    def get_random_nrn_inaccessible_stability(self, p, max_s=10, randgauss=False):

        # memory load parameters
        self.p = p
        self.P = int(self.s * self.p)
        self.W = torch.zeros((self.N, self.N), device=self.device)
        s_sample = min(self.s, max_s)  # nr of contexts to explicitly store
        s_extra = max(0, self.s - s_sample)  # extra contexts (not tested)
        P_sample = s_sample * self.p
        P_extra = s_extra * self.p

        # first incorporate extra contexts which won't be tested
        if s_extra > 0:
            nrn_mask = torch.repeat_interleave(
                                torch.bernoulli(self.a * torch.ones((s_extra, self.N), device=self.device)).bool(),
                                self.p * torch.ones((s_extra,), dtype=bool, device=self.device), dim=0)
            SXi = nrn_mask * (-1 + 2 * torch.bernoulli(0.5 * torch.ones((P_extra, self.N), device=self.device)))
            self.W += self.eta * torch.t(SXi) @ SXi
            del nrn_mask, SXi

        # now define contexts which will be tested, save nrn gating and patterns
        nrn_gating = torch.bernoulli(self.a * torch.ones((s_sample, self.N), device=self.device)).bool()
        nrn_mask = torch.repeat_interleave(nrn_gating,
                                           self.p * torch.ones((s_sample,), dtype=bool, device=self.device), dim=0)
        SXi = nrn_mask * (-1 + 2 * torch.bernoulli(0.5 * torch.ones((P_sample, self.N), device=self.device)))
        del nrn_mask
        self.W += self.eta * torch.t(SXi) @ SXi
        self.W.as_strided([self.N], [self.N + 1]).copy_(torch.diag(self.W) * 0.0)

        if randgauss:
            self.W = torch.normal(mean=torch.zeros((self.Ncxt, self.Ncxt), device=self.device))
            self.W.as_strided([self.Ncxt], [self.Ncxt + 1]).copy_(torch.diag(self.W) * 0.0)

        # loop through subnetworks and calculate stability of other memory patterns
        avgstab, stdstab, avgstab_nonzero, stdstab_nonzero = [], [], [], []
        for i in range(s_sample):
            Wtmp = self.W.clone()
            patinds = [x for x in range(0, i * self.p)] + [x for x in range((i + 1) * self.p, P_sample)]

            if self.syn_ref:
                W0 = torch.t(SXi[i * self.p:(i+1) * self.p, :]) @ SXi[i * self.p:(i+1) * self.p, :]
                M = (W0 * Wtmp) + 0 * torch.normal(mean=torch.zeros((self.N, self.N), device=self.device)) < 0.
                Wtmp[M] = 0.
                del M

            Wtmp *= torch.outer(nrn_gating[i, :], nrn_gating[i, :])
            Wtmp.as_strided([self.N], [self.N + 1]).copy_(torch.diag(Wtmp) * 0.0)

            stability = torch.div(SXi[patinds, :] * (SXi[patinds, :] @ Wtmp), torch.norm(Wtmp, dim=0))
            avgstab.append(stability[~torch.isnan(stability)].mean().item())
            stdstab.append(stability[~torch.isnan(stability)].std().item())
            stability[torch.abs(stability) < 1e-6] = np.nan
            avgstab_nonzero.append(stability[~torch.isnan(stability)].mean().item())
            stdstab_nonzero.append(stability[~torch.isnan(stability)].std().item())
            del Wtmp, stability
        del SXi, nrn_gating
        return np.mean(avgstab), np.mean(stdstab), np.mean(avgstab_nonzero), np.mean(stdstab_nonzero)

    def get_random_syn_inaccessible_stability(self, p, max_s=10):

        # memory load parameters
        self.p = p
        self.P = int(self.s * self.p)
        self.W = torch.zeros((self.N, self.N), device=self.device)
        s_sample = min(self.s, max_s)  # nr of contexts to explicitly store
        s_extra = max(0, self.s - s_sample)  # extra contexts (not tested)
        P_sample = s_sample * self.p
        P_extra = s_extra * self.p
        self.initialize_weights()

        # first incorporate extra contexts which won't be tested
        for i in range(s_extra):
            Xi = -1 + 2 * torch.bernoulli(0.5 * torch.ones((self.p, self.N), device=self.device))
            half_syn_mask = torch.bernoulli(np.sqrt(self.c) * torch.ones((self.N, self.N), device=self.device)).bool()
            syn_mask = half_syn_mask * torch.t(half_syn_mask)
            self.W += self.eta * torch.mul(syn_mask, torch.matmul(torch.t(Xi), Xi))
            del half_syn_mask, syn_mask, Xi

        # now define contexts which will be tested, save nrn gating and patterns
        syn_gating = torch.zeros((s_sample, self.N, self.N), device=self.device, dtype=bool)
        SXi = torch.zeros((P_sample, self.N), device=self.device)
        for i in range(s_sample):
            SXi[(i * self.p):((i + 1) * self.p), :] = -1 + 2 * torch.bernoulli(0.5 * torch.ones((self.p, self.N),
                                                                                                device=self.device))
            half_syn_mask = torch.bernoulli(np.sqrt(self.c) * torch.ones((self.N, self.N), device=self.device)).bool()
            syn_gating[i, :, :] = half_syn_mask * torch.t(half_syn_mask)
            del half_syn_mask
            self.W += self.eta * syn_gating[i, :, :] * (torch.t(SXi[(i * self.p):((i + 1) * self.p), :])
                                                        @ SXi[(i * self.p):((i + 1) * self.p), :])

        self.W.as_strided([self.N], [self.N + 1]).copy_(torch.diag(self.W) * 0.0)

        # loop through subnetworks and calculate stability of other memory patterns
        avgstab, stdstab, avgstab_nonzero, stdstab_nonzero = [], [], [], []
        for i in range(s_sample):
            Wtmp = torch.mul(syn_gating[i, :, :], self.W)
            patinds = [x for x in range(0, i * self.p)] + [x for x in range((i + 1) * self.p, P_sample)]
            stability = torch.div(SXi[patinds, :] * (SXi[patinds, :] @ Wtmp), torch.norm(Wtmp, dim=0))
            avgstab.append(stability[~torch.isnan(stability)].mean().item())
            stdstab.append(stability[~torch.isnan(stability)].std().item())
            stability[torch.abs(stability) < 1e-9] = np.nan
            avgstab_nonzero.append(stability[~torch.isnan(stability)].mean().item())
            stdstab_nonzero.append(stability[~torch.isnan(stability)].std().item())
            del Wtmp, stability
        del SXi, syn_gating
        return np.mean(avgstab), np.mean(stdstab), np.mean(avgstab_nonzero), np.mean(stdstab_nonzero)

    def clear_memory(self):
        try:
            del self.Xi
        except AttributeError:
            pass
        try:
            del self.W
        except AttributeError:
            pass
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        self.Xi, self.W = None, None
