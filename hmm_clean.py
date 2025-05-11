import numpy as np
import itertools
import json
from utils import np_multivariate_normal_pdf, convertcls_vect, calc_matDS, convert_multcls_vectors, calc_transDS
from scipy.stats import multivariate_normal
from scipy.special import logsumexp
from sklearn.cluster import KMeans


class HMC_ctod:
    __slots__ = ('p', 't', 'mu', 'sigma', 'nbc_x', 'vect', 'reg')

    def __init__(self, nbc_x, p=None, t=None, mu=None, sigma=None, vect=True, reg=10**-10, obs_indep=False):
        self.p = p
        self.t = t
        self.mu = mu
        self.sigma = sigma
        self.nbc_x = nbc_x
        self.vect = vect
        self.reg = reg

    def init_data_prior(self, data, scale=1):
        nb_class = self.nbc_x
        self.p = np.array([1 / nb_class] * nb_class)
        a = np.full((nb_class, nb_class), 1 / (2 * (nb_class - 1)))
        a = a - np.diag(np.diag(a))
        self.t = np.diag(np.array([1 / 2] * nb_class)) + a
        M = np.mean(data, axis=0)
        Sig = np.cov(data, rowvar=False).reshape(data.shape[1], data.shape[1])
        self.mu = [None] * nb_class
        self.sigma = [None] * nb_class
        for l in range(nb_class):
            if l % 2 == 0:
                self.mu[l] = M - scale * ((l / (nb_class / 2)) * np.sum(Sig, axis=1))
            else:
                self.mu[l] = M + scale * ((l / (nb_class / 2)) * np.sum(Sig, axis=1))
            self.sigma[l] = Sig

        self.mu = np.array(self.mu)
        self.sigma = np.array(self.sigma)

    def init_kmeans(self, data):
        self.t = np.zeros((self.nbc_x, self.nbc_x))
        self.p = np.zeros((self.nbc_x,))
        self.mu = np.zeros((self.nbc_x, 1 * len(data[0])))
        self.sigma = np.zeros((self.nbc_x, 1 * len(data[0]), 1 * len(data[0])))

        kmeans = KMeans(n_clusters=self.nbc_x).fit(data)
        hidden = kmeans.labels_
        hiddenc = np.stack((hidden[:-1], hidden[1:]), axis=-1)
        aux = np.moveaxis(np.indices(self.t.shape), 0, -1)
        broadc = (len(aux.shape) - len(hiddenc.shape) + 1)

        c = (1 / (len(data) - 1)) * (
            np.all(hiddenc.reshape((hiddenc.shape[0],) + (1,) * broadc + hiddenc.shape[1:]) == aux, axis=-1).sum(
                axis=0)).reshape(self.t.shape)

        self.p = (1 / (len(data))) * (hidden[..., np.newaxis] == np.indices((self.p.shape[0],))).sum(axis=0)
        self.t = (c.T / self.p).T

        self.mu = (((hidden[..., np.newaxis] == np.indices((self.mu.shape[0],)))[..., np.newaxis] * data[:, np.newaxis,
                                                                                                    ...]).sum(axis=0) /
                   (
                           hidden[..., np.newaxis] == np.indices((self.mu.shape[0],))).sum(axis=0)[
                       ..., np.newaxis]).reshape(self.mu.shape)
        self.sigma = (((hidden[..., np.newaxis] == np.indices((self.mu.shape[0],))).reshape(
            (hidden.shape[0], self.mu.shape[0]))[..., np.newaxis, np.newaxis] * np.einsum('...i,...j',
                                                                                          (data[:, np.newaxis, ...] -
                                                                                           self.mu[np.newaxis, ...]),
                                                                                          (data[:, np.newaxis, ...] -
                                                                                           self.mu[
                                                                                               np.newaxis, ...]))).sum(
            axis=0)
                      / ((hidden[..., np.newaxis] == np.indices((self.mu.shape[0],))).sum(
                    axis=0)).reshape((self.mu.shape[0],))[..., np.newaxis, np.newaxis])

    def give_param(self, c, mu, sigma):
        self.p = np.sum(c, axis=1)
        self.t = (c.T / self.p).T
        self.t[np.isnan(self.t)] = 0
        self.mu = mu
        self.sigma = sigma

    def get_param_give_form(self):
        c = (self.t.T * self.p).T
        return c, self.mu, self.sigma

    def get_param_apri_sup(self, i,j):
        if i is not None:
            p = np.zeros(self.p.shape)
            p[i] = self.p[i]
            if j is not None:
                t = np.zeros(self.t.shape)
                t[i, j] = self.t[i, j]
            else:
                t = np.zeros(self.t.shape)
                t[i, :] = self.t[i, :]
        elif j is not None:
            t = np.zeros(self.t.shape)
            t[:, j] = self.t[:, j]
            p = self.p
        else:
            p = self.p
            t = self.t
        return p, t

    def get_all_params_apri_sup(self, hidden):
        params = [self.get_param_apri_sup(hidden[i],hidden[i+1]) for i in range(hidden.shape[0]-1)]
        P = np.array([ele[0] for ele in params] + [self.get_param_apri_sup(hidden[-1],None)[0]])
        T = np.array([ele[1] for ele in params])
        return P,T

    def save_param_to_json(self, filepath):
        param_s = {'p': self.p.tolist(), 't': self.t.tolist(), 'mu': self.mu.tolist(),
                   'sig': self.sigma.tolist()}
        with open(filepath,'w') as f:
            json.dump(param_s, f, ensure_ascii=False)

    def load_param_from_json(self, filepath):
        with open(filepath, 'r') as f:
            params = json.load(f)
        self.p = np.array(params['p'])
        self.t = np.array(params['t'])
        self.mu = np.array(params['mu'])
        self.sigma = np.array(params['sig'])

    def seg_map(self, data, hidden=None):
        pass

    def seg_mpm(self, data, hidden=None):
        gaussians = np_multivariate_normal_pdf(data, self.mu, self.sigma)
        forward, backward = self.get_forward_backward(gaussians, hidden=hidden)
        if self.vect:
            p_apost = forward * backward
            p_apost = p_apost / (p_apost.sum(axis=1)[..., np.newaxis])
            return np.argmax(p_apost, axis=1)
        else:
            res = np.zeros((data.shape[0]))
            for i in range(len(res)):
                p_apost_i = forward[i] * backward[i]
                p_apost_i = p_apost_i / p_apost_i.sum()
                res[i] = np.argmax(p_apost_i)
            return res

    def simul_hidden_apost(self, backward, gaussians, hidden=None):
        res = np.zeros(len(backward), dtype=int)
        T = self.t
        aux = (gaussians[0] * self.p) * backward[0]
        test = np.random.multinomial(1, aux / np.sum(aux))
        res[0] = np.argmax(test)
        if self.vect:
            T = T[np.newaxis, :, :]
            if hidden is not None:
                P, T = self.get_all_params_apri_sup(hidden)
            tapost = (
                (gaussians[1:, np.newaxis, :]
                 * backward[1:, np.newaxis, :]
                 * T)
            )
            tapost = tapost / tapost.sum(axis=2)[..., np.newaxis]
            for i in range(1, len(res)):
                test = np.random.multinomial(1, tapost[i - 1, res[i - 1], :])
                res[i] = np.argmax(test)
        else:
            for i in range(1, len(res)):
                if hidden is not None:
                    P, T = self.get_param_apri_sup(hidden[i-1], hidden[i])
                tapost_i = (
                    (gaussians[i, np.newaxis, :]
                     * backward[i, np.newaxis, :]
                     * T[:, :])
                )
                tapost_i = tapost_i / tapost_i.sum(axis=1)[..., np.newaxis]
                test = np.random.multinomial(1, tapost_i[res[i - 1], :])
                res[i] = np.argmax(test)
        return res

    def simul_visible(self, hidden):
        res = np.zeros((len(hidden), self.mu.shape[-1]))
        for i in range(0, len(res)):
            res[i] = multivariate_normal.rvs(self.mu[hidden[i]], self.sigma[hidden[i]])
        return res

    def generate_sample(self, length):
        hidden = np.zeros(length, dtype=int)
        visible = np.zeros((length, self.mu.shape[-1]))
        T = self.t
        test = np.random.multinomial(1, self.p)
        hidden[0] = np.argmax(test)
        visible[0] = multivariate_normal.rvs(self.mu[hidden[0]], self.sigma[hidden[0]])
        for i in range(1, length):
            test = np.random.multinomial(1, T[hidden[i - 1], :])
            hidden[i] = np.argmax(test)
            visible[i] = multivariate_normal.rvs(self.mu[hidden[i]], self.sigma[hidden[i]])

        return hidden, visible

    def get_forward_backward(self, gaussians, log_forward=False, hidden=None):
        P = self.p
        T = self.t
        if hidden is not None:
            P,T = self.get_param_apri_sup(hidden[0],hidden[1])
        forward = np.zeros((len(gaussians), self.t.shape[0]))
        backward = np.zeros((len(gaussians), self.t.shape[0]))
        backward[len(gaussians) - 1] = np.ones(self.t.shape[0])
        forward[0] = P * gaussians[0]
        forward[0] = forward[0] / (forward[0].sum())
        if not log_forward:
            for l in range(1, len(gaussians)):
                k = len(gaussians) - 1 - l
                if hidden is not None:
                    P, T = self.get_param_apri_sup(hidden[l-1], hidden[l])
                forward[l] = gaussians[l] * (forward[l - 1] @ T)
                forward[l] = forward[l] / forward[l].sum()
                if hidden is not None:
                    P, T = self.get_param_apri_sup(hidden[k], hidden[k+1])
                backward[k] = (gaussians[k + 1] * backward[k + 1]) @ T.T
                backward[k] = backward[k] / (backward[k].sum())

            return forward, backward
        else:
            log_forward = np.zeros((len(gaussians), self.t.shape[0]))
            log_forward[0] = np.log(self.p * gaussians[0])
            for l in range(1, len(gaussians)):
                k = len(gaussians) - 1 - l
                if hidden is not None:
                    P, T = self.get_param_apri_sup(hidden[l-1], hidden[l])
                log_forward[l] = logsumexp((np.log(T.T) + log_forward[l - 1]).T + np.log(
                    gaussians[l]), axis=0)
                # log_forward[l] = np.logaddexp(logsumexp(np.logaddexp(np.log(self.t.T),log_forward[l - 1][np.newaxis,...]).T,axis=0), np.log(gaussians[l]))
                # log_forward[l] = np.apply_along_axis(ln_sum_np, 0, (np.log(self.t.T) + log_forward[l - 1]).T + np.log(
                #     gaussians[l]))
                forward[l] = gaussians[l] * (forward[l - 1] @ T)
                forward[l] = forward[l] / forward[l].sum()
                if hidden is not None:
                    P, T = self.get_param_apri_sup(hidden[k], hidden[k+1])
                backward[k] = (gaussians[k + 1] * backward[k + 1]) @ T.T
                backward[k] = backward[k] / (backward[k].sum())

            return forward, backward, log_forward

    def calc_param_EM(self, data, forward, backward, gaussians, hidden=None):
        T = self.t
        if self.vect:
            T = T[np.newaxis, :, :]
            if hidden is not None:
                P, T = self.get_all_params_apri_sup(hidden)
            gamma = (
                    forward[:-1, :, np.newaxis]
                    * (gaussians[1:, np.newaxis, :]
                       * backward[1:, np.newaxis, :]
                       * T)
            )
            gamma = gamma / (gamma.sum(axis=(1, 2))[..., np.newaxis, np.newaxis])
            psi = forward * backward
            psi = psi / (psi.sum(axis=1)[..., np.newaxis])
            aux = (psi[:-1:].sum(axis=0))
            aux[aux == 0] = 1 * 10 ** -100
            self.t = np.transpose(np.transpose((gamma.sum(axis=0))) / aux)
            self.p = (psi.sum(axis=0)) / psi.shape[0]
            self.mu = (((psi[..., np.newaxis] * data[:, np.newaxis, ...]).sum(axis=0)) / (
                psi.sum(axis=0)[..., np.newaxis])).reshape(self.mu.shape)

            self.sigma = (psi.reshape((psi.shape[0], self.mu.shape[0]))[..., np.newaxis, np.newaxis] * np.einsum(
                '...i,...j',
                (data[:, np.newaxis, ...] - self.mu[np.newaxis, ...]),
                (data[:, np.newaxis, ...] -
                 self.mu[np.newaxis, ...])
            )).sum(
                axis=0) / (psi.sum(axis=0)[..., np.newaxis, np.newaxis])
        else:
            psi = (forward[0] * backward[0]) / (forward[0] * backward[0]).sum()
            gamma = np.zeros(T.shape)
            aux = np.zeros(self.p.shape)
            mu_aux = (psi[..., np.newaxis] * data[0, np.newaxis, ...])
            sigma_aux = np.zeros(self.sigma.shape)
            for i in range(1, len(gaussians)):
                if hidden is not None:
                    P, T = self.get_param_apri_sup(hidden[i-1], hidden[i])
                aux = aux + (forward[i - 1] * backward[i - 1]) / (forward[i - 1] * backward[i - 1]).sum()
                psi = psi + (forward[i] * backward[i]) / (forward[i] * backward[i]).sum()
                gamma = gamma + (
                        forward[i - 1, :, np.newaxis]
                        * (gaussians[i, np.newaxis, :]
                           * backward[i, np.newaxis, :]
                           * T[:, :])
                ) / (
                                forward[i - 1, :, np.newaxis]
                                * (gaussians[i, np.newaxis, :]
                                   * backward[i, np.newaxis, :]
                                   * T[:, :])
                        ).sum()
                mu_aux = mu_aux + (
                        ((forward[i] * backward[i]) / (forward[i] * backward[i]).sum())[..., np.newaxis] * data[
                    i, np.newaxis, ...])
            self.t = np.transpose(np.transpose(gamma) / aux)
            self.p = psi / len(data)
            self.mu = (mu_aux / psi[..., np.newaxis]).reshape(self.mu.shape)
            for i in range(len(gaussians)):
                sigma_aux = sigma_aux + (((forward[i] * backward[i]) / (forward[i] * backward[i]).sum())[
                                             ..., np.newaxis, np.newaxis] * np.einsum('...i,...j',
                                                                                      (data[i, np.newaxis, ...] -
                                                                                       self.mu[np.newaxis, ...]),
                                                                                      (data[i, np.newaxis, ...] -
                                                                                       self.mu[np.newaxis, ...])
                                                                                      )).reshape(self.sigma.shape)
            self.sigma = (sigma_aux / psi[..., np.newaxis, np.newaxis])
        self.sigma = self.sigma+np.full(self.sigma.shape, np.eye(self.sigma.shape[-1]) * self.reg)

    def calc_param_ICE(self, data, forward, backward, gaussians, Nb_simul, hidden=None):
        s_hidden = np.stack([self.simul_hidden_apost(backward, gaussians, hidden=hidden) for n in range(Nb_simul)], axis=0)
        T = self.t
        if self.vect:
            T = T[np.newaxis, :, :]
            if hidden is not None:
                P, T = self.get_all_params_apri_sup(hidden)
            gamma = (
                    forward[:-1, :, np.newaxis]
                    * (gaussians[1:, np.newaxis, :]
                       * backward[1:, np.newaxis, :]
                       * T)
            )
            gamma = gamma / (gamma.sum(axis=(1, 2))[..., np.newaxis, np.newaxis])
            psi = forward * backward
            psi = psi / (psi.sum(axis=1)[..., np.newaxis])
            aux = (psi[:-1:].sum(axis=0))
            aux[aux == 0] = 1 * 10 ** -100
            self.t = np.transpose(np.transpose((gamma.sum(axis=0))) / aux)
            self.p = (psi.sum(axis=0)) / psi.shape[0]

            self.mu = (((s_hidden[..., np.newaxis] == np.indices((self.mu.shape[0],)))[..., np.newaxis] * data[:,
                                                                                                        np.newaxis,
                                                                                                        ...]).sum(
                axis=(0, 1)) / (
                               s_hidden[..., np.newaxis] == np.indices((self.mu.shape[0],))).sum(axis=(0, 1))[
                           ..., np.newaxis]).reshape(self.mu.shape)
            self.sigma = (((s_hidden[..., np.newaxis] == np.indices((self.mu.shape[0],))).reshape(
                (s_hidden.shape[0], hidden.shape[1], self.mu.shape[0]))[..., np.newaxis, np.newaxis] * np.einsum(
                '...i,...j',
                (data[:, np.newaxis,
                 ...] - self.mu[
                     np.newaxis, ...]),
                (data[:, np.newaxis,
                 ...] -
                 self.mu[
                     np.newaxis, ...]))).sum(
                axis=(0, 1))
                          / ((s_hidden[..., np.newaxis] == np.indices((self.mu.shape[0],))).sum(
                        axis=(0, 1))).reshape((self.mu.shape[0],))[..., np.newaxis, np.newaxis])
        else:
            psi = (forward[0] * backward[0]) / (forward[0] * backward[0]).sum()
            psi_prime = (hidden[:, 0, np.newaxis] == np.indices((self.mu.shape[0],))).sum(axis=0)
            gamma = np.zeros(T.shape)
            aux = np.zeros(self.p.shape)
            mu_aux = ((hidden[..., 0, np.newaxis] == np.indices((self.mu.shape[0],))) * data[0]).sum(axis=0)
            sigma_aux = np.zeros(self.sigma.shape)
            for i in range(1, len(gaussians)):
                if hidden is not None:
                    P, T = self.get_param_apri_sup(hidden[i-1], hidden[i])
                aux = aux + (forward[i - 1] * backward[i - 1]) / (forward[i - 1] * backward[i - 1]).sum()
                psi = psi + (forward[i] * backward[i]) / (forward[i] * backward[i]).sum()
                gamma = gamma + (
                        forward[i - 1, :, np.newaxis]
                        * (gaussians[i, np.newaxis, :]
                           * backward[i, np.newaxis, :]
                           * T[:, :])
                ) / (
                                forward[i - 1, :, np.newaxis]
                                * (gaussians[i, np.newaxis, :]
                                   * backward[i, np.newaxis, :]
                                   * T[:, :])
                        ).sum()
                psi_prime = psi_prime + (s_hidden[..., i, np.newaxis] == np.indices((self.mu.shape[0],))).sum(axis=0)
                mu_aux = mu_aux + ((s_hidden[..., i, np.newaxis] == np.indices((self.mu.shape[0],))) * data[i]).sum(
                    axis=0)
            self.t = np.transpose(np.transpose(gamma) / aux)
            self.p = psi / len(data)
            self.mu = (mu_aux / psi_prime[..., np.newaxis]).reshape(self.mu.shape)
            for i in range(len(gaussians)):
                sigma_aux = sigma_aux + ((s_hidden[..., i, np.newaxis] == np.indices((self.mu.shape[0],)))[
                                             ..., np.newaxis, np.newaxis] * np.einsum('...i,...j',
                                                                                      (data[i, np.newaxis, ...] -
                                                                                       self.mu[np.newaxis, ...]),
                                                                                      (data[i, np.newaxis, ...] -
                                                                                       self.mu[np.newaxis, ...])
                                                                                      )).sum(axis=0).reshape(
                    self.sigma.shape)
            self.sigma = (sigma_aux / psi_prime[..., np.newaxis, np.newaxis])
        self.sigma = self.sigma+np.full(self.sigma.shape, np.eye(self.sigma.shape[-1]) * self.reg)

    def calc_param_SEM(self, data, forward, backward, gaussians, hidden=None, pairs=False):
        hidden = self.simul_hidden_apost(backward, gaussians, hidden=hidden)
        hiddenc = np.stack((hidden[:-1], hidden[1:]), axis=-1)
        aux = np.moveaxis(np.indices(self.t.shape), 0, -1)
        if self.vect:
            c = (1 / (len(data) - 1)) * (
                np.all(hiddenc[:, np.newaxis, np.newaxis, ...] == aux[np.newaxis, ...], axis=-1).sum(
                    axis=0))
            self.p = (1 / (len(data))) * (hidden[..., np.newaxis] == np.indices((self.p.shape[0],))).sum(axis=0)
            self.t = (c.T / self.p).T
            self.t[np.isnan(self.t)] = 0
            self.mu = (((hidden[..., np.newaxis] == np.indices((self.mu.shape[0],)))[..., np.newaxis] * data[:,
                                                                                                        np.newaxis,
                                                                                                        ...]).sum(
                axis=(0)) / (
                               hidden[..., np.newaxis] == np.indices((self.mu.shape[0],))).sum(axis=(0))[
                           ..., np.newaxis]).reshape(self.mu.shape)
            self.sigma = (((hidden[..., np.newaxis] == np.indices((self.mu.shape[0],))).reshape(
                (hidden.shape[0], self.mu.shape[0]))[..., np.newaxis, np.newaxis] * np.einsum(
                '...i,...j',
                (data[:, np.newaxis,
                 ...] - self.mu[
                     np.newaxis, ...]),
                (data[:, np.newaxis,
                 ...] -
                 self.mu[
                     np.newaxis, ...]))).sum(
                axis=(0))
                          / ((hidden[..., np.newaxis] == np.indices((self.mu.shape[0],))).sum(
                        axis=(0))).reshape((self.mu.shape[0],))[..., np.newaxis, np.newaxis]).reshape(
                self.sigma.shape)
        else:
            c = np.zeros(self.t.shape)
            p = (hidden[0, np.newaxis] == np.indices((self.p.shape[0],))).sum(axis=0)
            psi_prime = (hidden[0, np.newaxis] == np.indices((self.mu.shape[0],)))
            mu_aux = ((hidden[0, np.newaxis] == np.indices((self.mu.shape[0],))) * data[0])
            sigma_aux = np.zeros(self.sigma.shape)
            for i in range(1, len(data)):
                c = c + (
                    np.all(hiddenc[i - 1, np.newaxis, np.newaxis, ...] == aux[np.newaxis, ...], axis=-1)).sum(
                    axis=0)
                p = p + (hidden[i, np.newaxis] == np.indices((self.p.shape[0],))).sum(axis=0)
                psi_prime = psi_prime + (hidden[i, np.newaxis] == np.indices((self.mu.shape[0],))).sum(axis=0)
                mu_aux = mu_aux + ((hidden[i, np.newaxis] == np.indices((self.mu.shape[0],))) * data[i]).sum(axis=0)
            self.p = (1 / (len(data))) * p
            c = (1 / (len(data) - 1)) * c
            self.t = (c.T / self.p).T
            self.mu = (mu_aux / psi_prime[..., np.newaxis]).reshape(self.mu.shape)
            for i in range(len(data)):
                sigma_aux = sigma_aux + ((hidden[i, np.newaxis] == np.indices((self.mu.shape[0],)))[
                                             ..., np.newaxis, np.newaxis] * np.einsum('...i,...j',
                                                                                      (data[i, np.newaxis, ...] -
                                                                                       self.mu[np.newaxis, ...]),
                                                                                      (data[i, np.newaxis, ...] -
                                                                                       self.mu[np.newaxis, ...])
                                                                                      )).sum(axis=0).reshape(
                    self.sigma.shape)
            self.sigma = (sigma_aux / psi_prime[..., np.newaxis, np.newaxis])
        self.sigma = self.sigma+np.full(self.sigma.shape, np.eye(self.sigma.shape[-1]) * self.reg)

    def get_param_EM(self, data, iter, early_stopping=0, true_ll=False, hidden=None):
        print({'iter': 0, 'p': self.p, 't': self.t, 'mu': self.mu, 'sigma': self.sigma})
        prev_log_likelihood = 0
        for q in range(iter):
            gaussians = np_multivariate_normal_pdf(data, self.mu, self.sigma)
            if true_ll:
                forward, backward, log_forward = self.get_forward_backward(gaussians, log_forward=True, hidden=hidden)
                new_log_likelihood = logsumexp(log_forward[-1])
            else:
                forward, backward = self.get_forward_backward(gaussians, hidden=hidden)
                prev_p = self.p
                prev_t = self.t
                prev_mu = self.mu
                prev_sigma = self.sigma
            self.calc_param_EM(data, forward, backward, gaussians, hidden=hidden)

            if early_stopping is not None:
                if true_ll:
                    diff_log_likelihood = np.abs((new_log_likelihood - prev_log_likelihood) / new_log_likelihood)
                    prev_log_likelihood = new_log_likelihood
                else:
                    diff_log_likelihood =  np.sqrt(
                ((self.p - prev_p) ** 2).sum() + ((self.t - prev_t) ** 2).sum() + ((self.mu - prev_mu) ** 2).sum() + (
                        (self.sigma - prev_sigma) ** 2).sum())
                print({'iter': q + 1, 'p': self.p, 't': self.t, 'mu': self.mu, 'sigma': self.sigma,
                       'diff_norm_param': diff_log_likelihood})
                if diff_log_likelihood <= early_stopping:
                    break
            else:
                print({'iter': q + 1, 'p': self.p, 't': self.t, 'mu': self.mu, 'sigma': self.sigma})

    def get_param_ICE(self, data, iter, Nb_simul, early_stopping=0, true_ll=False, hidden=None):
        print({'iter': 0, 'p': self.p, 't': self.t, 'mu': self.mu, 'sigma': self.sigma})
        prev_log_likelihood = 0
        for q in range(iter):
            gaussians = np_multivariate_normal_pdf(data, self.mu, self.sigma)
            if true_ll:
                forward, backward, log_forward = self.get_forward_backward(gaussians, log_forward=True, hidden=hidden)
                new_log_likelihood = logsumexp(log_forward[-1])
            else:
                forward, backward = self.get_forward_backward(gaussians, hidden=hidden)
                prev_p = self.p
                prev_t = self.t
                prev_mu = self.mu
                prev_sigma = self.sigma
            self.calc_param_ICE(data, forward, backward, gaussians, Nb_simul, hidden=hidden)

            if early_stopping is not None:
                if true_ll:
                    diff_log_likelihood = np.abs((new_log_likelihood - prev_log_likelihood) / new_log_likelihood)
                    prev_log_likelihood = new_log_likelihood
                else:
                    diff_log_likelihood = np.sqrt(
                        ((self.p - prev_p) ** 2).sum() + ((self.t - prev_t) ** 2).sum() + (
                                    (self.mu - prev_mu) ** 2).sum() + (
                                (self.sigma - prev_sigma) ** 2).sum())
                print({'iter': q + 1, 'p': self.p, 't': self.t, 'mu': self.mu, 'sigma': self.sigma,
                       'diff_norm_param': diff_log_likelihood})
                if diff_log_likelihood <= early_stopping:
                    break
            else:
                print({'iter': q + 1, 'p': self.p, 't': self.t, 'mu': self.mu, 'sigma': self.sigma})

    def get_param_SEM(self, data, iter, early_stopping=0, true_ll=False, hidden=None):
        print({'iter': 0, 'p': self.p, 't': self.t, 'mu': self.mu, 'sigma': self.sigma})
        prev_log_likelihood = 0
        for q in range(iter):
            gaussians = np_multivariate_normal_pdf(data, self.mu, self.sigma)
            if true_ll:
                forward, backward, log_forward = self.get_forward_backward(gaussians, log_forward=True, hidden=hidden)
                new_log_likelihood = logsumexp(log_forward[-1])
            else:
                forward, backward = self.get_forward_backward(gaussians, hidden=hidden)
                prev_p = self.p
                prev_t = self.t
                prev_mu = self.mu
                prev_sigma = self.sigma
            self.calc_param_SEM(data, forward, backward, gaussians, hidden=hidden)

            if early_stopping is not None:
                if true_ll:
                    diff_log_likelihood = np.abs((new_log_likelihood - prev_log_likelihood) / new_log_likelihood)
                    prev_log_likelihood = new_log_likelihood
                else:
                    diff_log_likelihood = np.sqrt(
                        ((self.p - prev_p) ** 2).sum() + ((self.t - prev_t) ** 2).sum() + (
                                (self.mu - prev_mu) ** 2).sum() + (
                                (self.sigma - prev_sigma) ** 2).sum())
                print({'iter': q + 1, 'p': self.p, 't': self.t, 'mu': self.mu, 'sigma': self.sigma,
                       'diff_norm_param': diff_log_likelihood})
                if diff_log_likelihood <= early_stopping:
                    break
            else:
                print({'iter': q + 1, 'p': self.p, 't': self.t, 'mu': self.mu, 'sigma': self.sigma})


class HMC_neigh_ctod:
    __slots__ = ('p', 't', 'mu', 'sigma', 'nbc_x', 'vect', 'reg')

    def __init__(self, nbc_x, p=None, t=None, mu=None, sigma=None, vect=True, reg=10**-10):
        self.p = p
        self.t = t
        self.mu = mu
        self.sigma = sigma
        self.nbc_x = nbc_x
        self.vect = vect
        self.reg=reg

    def init_data_prior(self, data, scale=1):
        nb_class = self.nbc_x
        self.p = np.array([1 / nb_class] * nb_class)
        a = np.full((nb_class, nb_class), 1 / (2 * (nb_class - 1)))
        a = a - np.diag(np.diag(a))
        self.t = np.diag(np.array([1 / 2] * nb_class)) + a
        M = np.mean(data, axis=0)
        Sig = np.cov(data, rowvar=False).reshape(data.shape[1], data.shape[1])
        self.mu = [None] * nb_class
        self.sigma = [None] * nb_class
        for l in range(nb_class):
            if l % 2 == 0:
                self.mu[l] = M - scale * ((l / (nb_class / 2)) * np.sum(Sig, axis=1))
            else:
                self.mu[l] = M + scale * ((l / (nb_class / 2)) * np.sum(Sig, axis=1))
            self.sigma[l] = Sig

        self.mu = np.array(self.mu)
        self.sigma = np.array(self.sigma)

    def init_kmeans(self, data):
        self.t = np.zeros((self.nbc_x, self.nbc_x))
        self.p = np.zeros((self.nbc_x,))
        self.mu = np.zeros((self.nbc_x, 1 * len(data[0])))
        self.sigma = np.zeros((self.nbc_x, 1 * len(data[0]), 1 * len(data[0])))

        kmeans = KMeans(n_clusters=self.nbc_x).fit(data)
        hidden = kmeans.labels_
        hiddenc = np.stack((hidden[:-1], hidden[1:]), axis=-1)
        aux = np.moveaxis(np.indices(self.t.shape), 0, -1)
        broadc = (len(aux.shape) - len(hiddenc.shape) + 1)

        c = (1 / (len(data) - 1)) * (
            np.all(hiddenc.reshape((hiddenc.shape[0],) + (1,) * broadc + hiddenc.shape[1:]) == aux, axis=-1).sum(
                axis=0)).reshape(self.t.shape)

        self.p = (1 / (len(data))) * (hidden[..., np.newaxis] == np.indices((self.p.shape[0],))).sum(axis=0)
        self.t = (c.T / self.p).T

        self.mu = (((hidden[..., np.newaxis] == np.indices((self.mu.shape[0],)))[..., np.newaxis] * data[:, np.newaxis,
                                                                                                    ...]).sum(axis=0) /
                   (
                           hidden[..., np.newaxis] == np.indices((self.mu.shape[0],))).sum(axis=0)[
                       ..., np.newaxis]).reshape(self.mu.shape)
        self.sigma = (((hidden[..., np.newaxis] == np.indices((self.mu.shape[0],))).reshape(
            (hidden.shape[0], self.mu.shape[0]))[..., np.newaxis, np.newaxis] * np.einsum('...i,...j',
                                                                                          (data[:, np.newaxis, ...] -
                                                                                           self.mu[np.newaxis, ...]),
                                                                                          (data[:, np.newaxis, ...] -
                                                                                           self.mu[
                                                                                               np.newaxis, ...]))).sum(
            axis=0)
                      / ((hidden[..., np.newaxis] == np.indices((self.mu.shape[0],))).sum(
                    axis=0)).reshape((self.mu.shape[0],))[..., np.newaxis, np.newaxis])

    def give_param(self, c, mu, sigma):
        self.p = np.sum(c, axis=1)
        self.t = (c.T / self.p).T
        self.t[np.isnan(self.t)] = 0
        self.mu = mu
        self.sigma = sigma

    def get_param_give_form(self):
        c = (self.t.T * self.p).T
        return c, self.mu, self.sigma

    def get_gaussians(self, data, data_neighh, data_neighv):
        mask_data = np.isnan(data)
        mask_neighh = np.isnan(data_neighh)
        mask_neighv = np.isnan(data_neighv)
        data[mask_data] = 0
        data_neighh[mask_neighh] = 0
        data_neighv[mask_neighv] = 0
        neighh = np.einsum('...ij,...j',self.t,np_multivariate_normal_pdf(data_neighh, self.mu, self.sigma)) + self.reg
        neighv = np.einsum('...ij,...j',self.t,np_multivariate_normal_pdf(data_neighv, self.mu, self.sigma)) + self.reg
        gausses = np_multivariate_normal_pdf(data, self.mu, self.sigma)
        neighh[np.any(mask_neighh, axis=1)] = 1
        neighv[np.any(mask_neighv, axis=1)] = 1
        gausses[np.any(mask_data, axis=1)] = 1
        return gausses*neighh*neighv

    def get_param_apri_sup(self, i,j):
        if i is not None:
            p = np.zeros(self.p.shape)
            p[i] = self.p[i]
            if j is not None:
                t = np.zeros(self.t.shape)
                t[i, j] = self.t[i, j]
            else:
                t = np.zeros(self.t.shape)
                t[i, :] = self.t[i, :]
        elif j is not None:
            t = np.zeros(self.t.shape)
            t[:, j] = self.t[:, j]
            p = self.p
        else:
            p = self.p
            t = self.t
        return p, t

    def get_all_params_apri_sup(self, hidden):
        params = [self.get_param_apri_sup(hidden[i],hidden[i+1]) for i in range(hidden.shape[0]-1)]
        P = np.array([ele[0] for ele in params] + [self.get_param_apri_sup(hidden[-1],None)[0]])
        T = np.array([ele[1] for ele in params])
        return P,T

    def save_param_to_json(self, filepath):
        param_s = {'p': self.p.tolist(), 't': self.t.tolist(), 'mu': self.mu.tolist(),
                   'sig': self.sigma.tolist()}
        with open(filepath,'w') as f:
            json.dump(param_s, f, ensure_ascii=False)

    def load_param_from_json(self, filepath):
        with open(filepath, 'r') as f:
            params = json.load(f)
        self.p = np.array(params['p'])
        self.t = np.array(params['t'])
        self.mu = np.array(params['mu'])
        self.sigma = np.array(params['sig'])

    def seg_map(self, data):
        pass

    def seg_mpm(self, data, data_neighh, data_neighv, hidden=None):
        gaussians = self.get_gaussians(data, data_neighh, data_neighv)
        forward, backward = self.get_forward_backward(gaussians, hidden=hidden)
        if self.vect:
            p_apost = forward * backward
            p_apost = p_apost / (p_apost.sum(axis=1)[..., np.newaxis])
            return np.argmax(p_apost, axis=1)
        else:
            res = np.zeros((data.shape[0]))
            for i in range(len(res)):
                p_apost_i = forward[i] * backward[i]
                p_apost_i = p_apost_i / p_apost_i.sum()
                res[i] = np.argmax(p_apost_i)
            return res

    def simul_hidden_apost(self, backward, gaussians, hidden=None):
        res = np.zeros(len(backward), dtype=int)
        T = self.t
        aux = (gaussians[0] * self.p) * backward[0]
        test = np.random.multinomial(1, aux / np.sum(aux))
        res[0] = np.argmax(test)
        if self.vect:
            T = T[np.newaxis, :, :]
            if hidden is not None:
                P, T = self.get_all_params_apri_sup(hidden)
            tapost = (
                (gaussians[1:, np.newaxis, :]
                 * backward[1:, np.newaxis, :]
                 * T)
            )
            tapost = tapost / tapost.sum(axis=2)[..., np.newaxis]
            for i in range(1, len(res)):
                test = np.random.multinomial(1, tapost[i - 1, res[i - 1], :])
                res[i] = np.argmax(test)
        else:
            for i in range(1, len(res)):
                if hidden is not None:
                    P, T = self.get_param_apri_sup(hidden[i-1], hidden[i])
                tapost_i = (
                    (gaussians[i, np.newaxis, :]
                     * backward[i, np.newaxis, :]
                     * T[:, :])
                )
                tapost_i = tapost_i / tapost_i.sum(axis=1)[..., np.newaxis]
                test = np.random.multinomial(1, tapost_i[res[i - 1], :])
                res[i] = np.argmax(test)
        return res

    def simul_visible(self, hidden):
        res = np.zeros((len(hidden), self.mu.shape[-1]))
        for i in range(0, len(res)):
            res[i] = multivariate_normal.rvs(self.mu[hidden[i]], self.sigma[hidden[i]])
        return res

    def generate_sample(self, length):
        hidden = np.zeros(length, dtype=int)
        visible = np.zeros((length, self.mu.shape[-1]))
        T = self.t
        test = np.random.multinomial(1, self.p)
        hidden[0] = np.argmax(test)
        visible[0] = multivariate_normal.rvs(self.mu[hidden[0]], self.sigma[hidden[0]])
        for i in range(1, length):
            test = np.random.multinomial(1, T[hidden[i - 1], :])
            hidden[i] = np.argmax(test)
            visible[i] = multivariate_normal.rvs(self.mu[hidden[i]], self.sigma[hidden[i]])

        return hidden, visible

    def get_forward_backward(self, gaussians, log_forward=False, hidden=None):
        P = self.p
        T = self.t
        if hidden is not None:
            P, T = self.get_param_apri_sup(hidden[0], hidden[1])
        forward = np.zeros((len(gaussians), self.t.shape[0]))
        backward = np.zeros((len(gaussians), self.t.shape[0]))
        backward[len(gaussians) - 1] = np.ones(self.t.shape[0])
        forward[0] = P * gaussians[0]
        forward[0] = forward[0] / (forward[0].sum())
        if not log_forward:
            for l in range(1, len(gaussians)):
                k = len(gaussians) - 1 - l
                if hidden is not None:
                    P, T = self.get_param_apri_sup(hidden[l - 1], hidden[l])
                forward[l] = gaussians[l] * (forward[l - 1] @ T)
                forward[l] = forward[l] / forward[l].sum()
                if hidden is not None:
                    P, T = self.get_param_apri_sup(hidden[k], hidden[k + 1])
                backward[k] = (gaussians[k + 1] * backward[k + 1]) @ T.T
                backward[k] = backward[k] / (backward[k].sum())

            return forward, backward
        else:
            log_forward = np.zeros((len(gaussians), self.t.shape[0]))
            log_forward[0] = np.log(self.p * gaussians[0])
            for l in range(1, len(gaussians)):
                k = len(gaussians) - 1 - l
                if hidden is not None:
                    P, T = self.get_param_apri_sup(hidden[l - 1], hidden[l])
                log_forward[l] = logsumexp((np.log(T.T) + log_forward[l - 1]).T + np.log(
                    gaussians[l]), axis=0)
                # log_forward[l] = np.logaddexp(logsumexp(np.logaddexp(np.log(self.t.T),log_forward[l - 1][np.newaxis,...]).T,axis=0), np.log(gaussians[l]))
                # log_forward[l] = np.apply_along_axis(ln_sum_np, 0, (np.log(self.t.T) + log_forward[l - 1]).T + np.log(
                #     gaussians[l]))
                forward[l] = gaussians[l] * (forward[l - 1] @ T)
                forward[l] = forward[l] / forward[l].sum()
                if hidden is not None:
                    P, T = self.get_param_apri_sup(hidden[k], hidden[k + 1])
                backward[k] = (gaussians[k + 1] * backward[k + 1]) @ T.T
                backward[k] = backward[k] / (backward[k].sum())

            return forward, backward, log_forward

    def calc_param_EM(self, data, forward, backward, gaussians, hidden=None):
        T = self.t
        if self.vect:
            T = T[np.newaxis, :, :]
            if hidden is not None:
                P, T = self.get_all_params_apri_sup(hidden)
            gamma = (
                    forward[:-1, :, np.newaxis]
                    * (gaussians[1:, np.newaxis, :]
                       * backward[1:, np.newaxis, :]
                       * T)
            )
            gamma = gamma / (gamma.sum(axis=(1, 2))[..., np.newaxis, np.newaxis])
            psi = forward * backward
            psi = psi / (psi.sum(axis=1)[..., np.newaxis])
            aux = (psi[:-1:].sum(axis=0))
            aux[aux == 0] = 1 * 10 ** -100
            self.t = np.transpose(np.transpose((gamma.sum(axis=0))) / aux)
            self.p = (psi.sum(axis=0)) / psi.shape[0]
            self.mu = (((psi[..., np.newaxis] * data[:, np.newaxis, ...]).sum(axis=0)) / (
                psi.sum(axis=0)[..., np.newaxis])).reshape(self.mu.shape)

            self.sigma = (psi.reshape((psi.shape[0], self.mu.shape[0]))[..., np.newaxis, np.newaxis] * np.einsum(
                '...i,...j',
                (data[:, np.newaxis, ...] - self.mu[np.newaxis, ...]),
                (data[:, np.newaxis, ...] -
                 self.mu[np.newaxis, ...])
            )).sum(
                axis=0) / (psi.sum(axis=0)[..., np.newaxis, np.newaxis])
        else:
            psi = (forward[0] * backward[0]) / (forward[0] * backward[0]).sum()
            gamma = np.zeros(T.shape)
            aux = np.zeros(self.p.shape)
            mu_aux = (psi[..., np.newaxis] * data[0, np.newaxis, ...])
            sigma_aux = np.zeros(self.sigma.shape)
            for i in range(1, len(gaussians)):
                if hidden is not None:
                    P, T = self.get_param_apri_sup(hidden[i-1], hidden[i])
                aux = aux + (forward[i - 1] * backward[i - 1]) / (forward[i - 1] * backward[i - 1]).sum()
                psi = psi + (forward[i] * backward[i]) / (forward[i] * backward[i]).sum()
                gamma = gamma + (
                        forward[i - 1, :, np.newaxis]
                        * (gaussians[i, np.newaxis, :]
                           * backward[i, np.newaxis, :]
                           * T[:, :])
                ) / (
                                forward[i - 1, :, np.newaxis]
                                * (gaussians[i, np.newaxis, :]
                                   * backward[i, np.newaxis, :]
                                   * T[:, :])
                        ).sum()
                mu_aux = mu_aux + (
                        ((forward[i] * backward[i]) / (forward[i] * backward[i]).sum())[..., np.newaxis] * data[
                    i, np.newaxis, ...])
            self.t = np.transpose(np.transpose(gamma) / aux)
            self.p = psi / len(data)
            self.mu = (mu_aux / psi[..., np.newaxis]).reshape(self.mu.shape)
            for i in range(len(gaussians)):
                sigma_aux = sigma_aux + (((forward[i] * backward[i]) / (forward[i] * backward[i]).sum())[
                                             ..., np.newaxis, np.newaxis] * np.einsum('...i,...j',
                                                                                      (data[i, np.newaxis, ...] -
                                                                                       self.mu[np.newaxis, ...]),
                                                                                      (data[i, np.newaxis, ...] -
                                                                                       self.mu[np.newaxis, ...])
                                                                                      )).reshape(self.sigma.shape)
            self.sigma = (sigma_aux / psi[..., np.newaxis, np.newaxis])
        self.sigma = self.sigma+np.full(self.sigma.shape, np.eye(self.sigma.shape[-1]) * self.reg)

    def calc_param_ICE(self, data, forward, backward, gaussians, Nb_simul, hidden=None):
        s_hidden = np.stack([self.simul_hidden_apost(backward, gaussians, hidden=hidden) for n in range(Nb_simul)], axis=0)
        T = self.t
        if self.vect:
            T = T[np.newaxis, :, :]
            if hidden is not None:
                P, T = self.get_all_params_apri_sup(hidden)
            gamma = (
                    forward[:-1, :, np.newaxis]
                    * (gaussians[1:, np.newaxis, :]
                       * backward[1:, np.newaxis, :]
                       * T)
            )
            gamma = gamma / (gamma.sum(axis=(1, 2))[..., np.newaxis, np.newaxis])
            psi = forward * backward
            psi = psi / (psi.sum(axis=1)[..., np.newaxis])
            aux = (psi[:-1:].sum(axis=0))
            aux[aux == 0] = 1 * 10 ** -100
            self.t = np.transpose(np.transpose((gamma.sum(axis=0))) / aux)
            self.p = (psi.sum(axis=0)) / psi.shape[0]

            self.mu = (((s_hidden[..., np.newaxis] == np.indices((self.mu.shape[0],)))[..., np.newaxis] * data[:,
                                                                                                        np.newaxis,
                                                                                                        ...]).sum(
                axis=(0, 1)) / (
                               s_hidden[..., np.newaxis] == np.indices((self.mu.shape[0],))).sum(axis=(0, 1))[
                           ..., np.newaxis]).reshape(self.mu.shape)
            self.sigma = (((s_hidden[..., np.newaxis] == np.indices((self.mu.shape[0],))).reshape(
                (s_hidden.shape[0], hidden.shape[1], self.mu.shape[0]))[..., np.newaxis, np.newaxis] * np.einsum(
                '...i,...j',
                (data[:, np.newaxis,
                 ...] - self.mu[
                     np.newaxis, ...]),
                (data[:, np.newaxis,
                 ...] -
                 self.mu[
                     np.newaxis, ...]))).sum(
                axis=(0, 1))
                          / ((s_hidden[..., np.newaxis] == np.indices((self.mu.shape[0],))).sum(
                        axis=(0, 1))).reshape((self.mu.shape[0],))[..., np.newaxis, np.newaxis])
        else:
            psi = (forward[0] * backward[0]) / (forward[0] * backward[0]).sum()
            psi_prime = (hidden[:, 0, np.newaxis] == np.indices((self.mu.shape[0],))).sum(axis=0)
            gamma = np.zeros(T.shape)
            aux = np.zeros(self.p.shape)
            mu_aux = ((hidden[..., 0, np.newaxis] == np.indices((self.mu.shape[0],))) * data[0]).sum(axis=0)
            sigma_aux = np.zeros(self.sigma.shape)
            for i in range(1, len(gaussians)):
                if hidden is not None:
                    P, T = self.get_param_apri_sup(hidden[i-1], hidden[i])
                aux = aux + (forward[i - 1] * backward[i - 1]) / (forward[i - 1] * backward[i - 1]).sum()
                psi = psi + (forward[i] * backward[i]) / (forward[i] * backward[i]).sum()
                gamma = gamma + (
                        forward[i - 1, :, np.newaxis]
                        * (gaussians[i, np.newaxis, :]
                           * backward[i, np.newaxis, :]
                           * T[:, :])
                ) / (
                                forward[i - 1, :, np.newaxis]
                                * (gaussians[i, np.newaxis, :]
                                   * backward[i, np.newaxis, :]
                                   * T[:, :])
                        ).sum()
                psi_prime = psi_prime + (s_hidden[..., i, np.newaxis] == np.indices((self.mu.shape[0],))).sum(axis=0)
                mu_aux = mu_aux + ((s_hidden[..., i, np.newaxis] == np.indices((self.mu.shape[0],))) * data[i]).sum(
                    axis=0)
            self.t = np.transpose(np.transpose(gamma) / aux)
            self.p = psi / len(data)
            self.mu = (mu_aux / psi_prime[..., np.newaxis]).reshape(self.mu.shape)
            for i in range(len(gaussians)):
                sigma_aux = sigma_aux + ((s_hidden[..., i, np.newaxis] == np.indices((self.mu.shape[0],)))[
                                             ..., np.newaxis, np.newaxis] * np.einsum('...i,...j',
                                                                                      (data[i, np.newaxis, ...] -
                                                                                       self.mu[np.newaxis, ...]),
                                                                                      (data[i, np.newaxis, ...] -
                                                                                       self.mu[np.newaxis, ...])
                                                                                      )).sum(axis=0).reshape(
                    self.sigma.shape)
            self.sigma = (sigma_aux / psi_prime[..., np.newaxis, np.newaxis])
        self.sigma = self.sigma+np.full(self.sigma.shape, np.eye(self.sigma.shape[-1]) * self.reg)

    def calc_param_SEM(self, data, forward, backward, gaussians, hidden=None):
        hidden = self.simul_hidden_apost(backward, gaussians, hidden=hidden)
        hiddenc = np.stack((hidden[:-1], hidden[1:]), axis=-1)
        aux = np.moveaxis(np.indices(self.t.shape), 0, -1)
        if self.vect:
            c = (1 / (len(data) - 1)) * (
                np.all(hiddenc[:, np.newaxis, np.newaxis, ...] == aux[np.newaxis, ...], axis=-1).sum(
                    axis=0))
            self.p = (1 / (len(data))) * (hidden[..., np.newaxis] == np.indices((self.p.shape[0],))).sum(axis=0)
            self.t = (c.T / self.p).T
            self.t[np.isnan(self.t)] = 0
            self.mu = (((hidden[..., np.newaxis] == np.indices((self.mu.shape[0],)))[..., np.newaxis] * data[:,
                                                                                                        np.newaxis,
                                                                                                        ...]).sum(
                axis=(0)) / (
                               hidden[..., np.newaxis] == np.indices((self.mu.shape[0],))).sum(axis=(0))[
                           ..., np.newaxis]).reshape(self.mu.shape)
            self.sigma = (((hidden[..., np.newaxis] == np.indices((self.mu.shape[0],))).reshape(
                (hidden.shape[0], self.mu.shape[0]))[..., np.newaxis, np.newaxis] * np.einsum(
                '...i,...j',
                (data[:, np.newaxis,
                 ...] - self.mu[
                     np.newaxis, ...]),
                (data[:, np.newaxis,
                 ...] -
                 self.mu[
                     np.newaxis, ...]))).sum(
                axis=(0))
                          / ((hidden[..., np.newaxis] == np.indices((self.mu.shape[0],))).sum(
                        axis=(0))).reshape((self.mu.shape[0],))[..., np.newaxis, np.newaxis]).reshape(
                self.sigma.shape)
        else:
            c = np.zeros(self.t.shape)
            p = (hidden[0, np.newaxis] == np.indices((self.p.shape[0],))).sum(axis=0)
            psi_prime = (hidden[0, np.newaxis] == np.indices((self.mu.shape[0],)))
            mu_aux = ((hidden[0, np.newaxis] == np.indices((self.mu.shape[0],))) * data[0])
            sigma_aux = np.zeros(self.sigma.shape)
            for i in range(1, len(data)):
                c = c + (
                    np.all(hiddenc[i - 1, np.newaxis, np.newaxis, ...] == aux[np.newaxis, ...], axis=-1)).sum(
                    axis=0)
                p = p + (hidden[i, np.newaxis] == np.indices((self.p.shape[0],))).sum(axis=0)
                psi_prime = psi_prime + (hidden[i, np.newaxis] == np.indices((self.mu.shape[0],))).sum(axis=0)
                mu_aux = mu_aux + ((hidden[i, np.newaxis] == np.indices((self.mu.shape[0],))) * data[i]).sum(axis=0)
            self.p = (1 / (len(data))) * p
            c = (1 / (len(data) - 1)) * c
            self.t = (c.T / self.p).T
            self.mu = (mu_aux / psi_prime[..., np.newaxis]).reshape(self.mu.shape)
            for i in range(len(data)):
                sigma_aux = sigma_aux + ((hidden[i, np.newaxis] == np.indices((self.mu.shape[0],)))[
                                             ..., np.newaxis, np.newaxis] * np.einsum('...i,...j',
                                                                                      (data[i, np.newaxis, ...] -
                                                                                       self.mu[np.newaxis, ...]),
                                                                                      (data[i, np.newaxis, ...] -
                                                                                       self.mu[np.newaxis, ...])
                                                                                      )).sum(axis=0).reshape(
                    self.sigma.shape)
            self.sigma = (sigma_aux / psi_prime[..., np.newaxis, np.newaxis])
        self.sigma = self.sigma+np.full(self.sigma.shape, np.eye(self.sigma.shape[-1]) * self.reg)

    def get_param_EM(self, data, data_neighh, data_neighv, iter, early_stopping=0, true_ll=False, hidden=None):
        print({'iter': 0, 'p': self.p, 't': self.t, 'mu': self.mu, 'sigma': self.sigma})
        prev_log_likelihood = 0
        for q in range(iter):
            gaussians = self.get_gaussians(data, data_neighh, data_neighv)
            if true_ll:
                forward, backward, log_forward = self.get_forward_backward(gaussians, log_forward=True, hidden=hidden)
                new_log_likelihood = logsumexp(log_forward[-1])
            else:
                forward, backward = self.get_forward_backward(gaussians, hidden=hidden)
                prev_p = self.p
                prev_t = self.t
                prev_mu = self.mu
                prev_sigma = self.sigma
            self.calc_param_EM(data, forward, backward, gaussians, hidden=hidden)

            if early_stopping is not None:
                if true_ll:
                    diff_log_likelihood = np.abs((new_log_likelihood - prev_log_likelihood) / new_log_likelihood)
                    prev_log_likelihood = new_log_likelihood
                else:
                    diff_log_likelihood = np.sqrt(
                        ((self.p - prev_p) ** 2).sum() + ((self.t - prev_t) ** 2).sum() + (
                                (self.mu - prev_mu) ** 2).sum() + (
                                (self.sigma - prev_sigma) ** 2).sum())
                print({'iter': q + 1, 'p': self.p, 't': self.t, 'mu': self.mu, 'sigma': self.sigma,
                       'diff_norm_param': diff_log_likelihood})
                if diff_log_likelihood <= early_stopping:
                    break
            else:
                print({'iter': q + 1, 'p': self.p, 't': self.t, 'mu': self.mu, 'sigma': self.sigma})

    def get_param_ICE(self, data, data_neighh, data_neighv, iter, Nb_simul, early_stopping=0, true_ll=False, hidden=None):
        print({'iter': 0, 'p': self.p, 't': self.t, 'mu': self.mu, 'sigma': self.sigma})
        prev_log_likelihood = 0
        for q in range(iter):
            gaussians = self.get_gaussians(data, data_neighh, data_neighv)
            if true_ll:
                forward, backward, log_forward = self.get_forward_backward(gaussians, log_forward=True, hidden=hidden)
                new_log_likelihood = logsumexp(log_forward[-1])
            else:
                forward, backward = self.get_forward_backward(gaussians, hidden=hidden)
                prev_p = self.p
                prev_t = self.t
                prev_mu = self.mu
                prev_sigma = self.sigma
            self.calc_param_ICE(data, forward, backward, gaussians, Nb_simul)

            if early_stopping is not None:
                if true_ll:
                    diff_log_likelihood = np.abs((new_log_likelihood - prev_log_likelihood) / new_log_likelihood)
                    prev_log_likelihood = new_log_likelihood
                else:
                    diff_log_likelihood = np.sqrt(
                        ((self.p - prev_p) ** 2).sum() + ((self.t - prev_t) ** 2).sum() + (
                                (self.mu - prev_mu) ** 2).sum() + (
                                (self.sigma - prev_sigma) ** 2).sum())
                print({'iter': q + 1, 'p': self.p, 't': self.t, 'mu': self.mu, 'sigma': self.sigma,
                       'diff_norm_param': diff_log_likelihood})
                if diff_log_likelihood <= early_stopping:
                    break
            else:
                print({'iter': q + 1, 'p': self.p, 't': self.t, 'mu': self.mu, 'sigma': self.sigma})

    def get_param_SEM(self, data, data_neighh, data_neighv, iter, early_stopping=0, true_ll=False, hidden=None):
        print({'iter': 0, 'p': self.p, 't': self.t, 'mu': self.mu, 'sigma': self.sigma})
        prev_log_likelihood = 0
        for q in range(iter):
            gaussians = self.get_gaussians(data, data_neighh, data_neighv)
            if true_ll:
                forward, backward, log_forward = self.get_forward_backward(gaussians, log_forward=True, hidden=hidden)
                new_log_likelihood = logsumexp(log_forward[-1])
            else:
                forward, backward = self.get_forward_backward(gaussians, hidden=hidden)
                prev_p = self.p
                prev_t = self.t
                prev_mu = self.mu
                prev_sigma = self.sigma
            self.calc_param_SEM(data, forward, backward, gaussians)

            if early_stopping is not None:
                if true_ll:
                    diff_log_likelihood = np.abs((new_log_likelihood - prev_log_likelihood) / new_log_likelihood)
                    prev_log_likelihood = new_log_likelihood
                else:
                    diff_log_likelihood = np.sqrt(
                        ((self.p - prev_p) ** 2).sum() + ((self.t - prev_t) ** 2).sum() + (
                                (self.mu - prev_mu) ** 2).sum() + (
                                (self.sigma - prev_sigma) ** 2).sum())
                print({'iter': q + 1, 'p': self.p, 't': self.t, 'mu': self.mu, 'sigma': self.sigma,
                       'diff_norm_param': diff_log_likelihood})
                if diff_log_likelihood <= early_stopping:
                    break
            else:
                print({'iter': q + 1, 'p': self.p, 't': self.t, 'mu': self.mu, 'sigma': self.sigma})


class HEMC_ctod:
    __slots__ = ('p', 't', 'lx', 'mu', 'sigma', 'nbc_x', 'nbc_u', 'vect', 'reg')

    def __init__(self, nbc_x, u='all', lx=None, p=None, t=None, mu=None, sigma=None, vect=True, reg=10**-10):
        self.reg = reg
        self.p = p
        self.t = t
        self.mu = mu
        self.sigma = sigma
        self.nbc_x = nbc_x
        if lx is None:
            if u == 'all':
                self.nbc_u = (2 ** nbc_x) - 1
            else:
                self.nbc_u = nbc_x + 1
        else:
            self.nbc_u = lx.shape[1]
        self.lx = lx
        self.vect = vect

    def init_data_prior(self, data, scale=1):
        if self.lx is None:
            if self.nbc_u == (2 ** self.nbc_x) - 1:
                self.lx = np.stack([convertcls_vect(i + 1, (2,) * self.nbc_x) for i in range(self.nbc_u)], axis=0).T
                card = 1 / np.sum(self.lx, axis=0)
                self.lx=self.lx*card
            else:
                self.lx = np.vstack((np.eye(self.nbc_x), np.ones((self.nbc_x,)))).T
                card = 1 / np.sum(self.lx, axis=0)
                self.lx = self.lx * card
        a = np.full((self.nbc_u, self.nbc_u), 1 / (2 * (self.nbc_u - 1)))
        a = a - np.diag(np.diag(a))
        p = np.array([1 / self.nbc_u] * self.nbc_u)
        t = np.diag(np.array([1 / 2] * self.nbc_u)) + a
        u = (t.T / p).T
        self.p = u.sum(axis=1)
        self.t = ((u).T / self.p).T

        M = np.mean(data, axis=0)
        Sig = np.cov(data, rowvar=False).reshape(data.shape[1], data.shape[1])
        self.mu = [None] * self.nbc_x
        self.sigma = [None] * self.nbc_x
        for l in range(self.nbc_x):
            if l % 2 == 0:
                self.mu[l] = M - scale * ((l / (self.nbc_x / 2)) * np.sum(Sig, axis=1))
            else:
                self.mu[l] = M + scale * ((l / (self.nbc_x / 2)) * np.sum(Sig, axis=1))
            self.sigma[l] = Sig

        self.mu = np.array(self.mu)
        self.sigma = np.array(self.sigma)

    def init_kmeans(self, data, perturbation_param=0.5):
        if self.lx is None:
            if self.nbc_u == (2 ** self.nbc_x) - 1:
                self.lx = np.stack([convertcls_vect(i + 1, (2,) * self.nbc_x) for i in range(self.nbc_u)], axis=0).T
                card = 1 / np.sum(self.lx, axis=0)
                self.lx=self.lx*card
            else:
                self.lx = np.vstack((np.eye(self.nbc_x), np.ones((self.nbc_x,)))).T
                card = 1 / np.sum(self.lx, axis=0)
                self.lx = self.lx * card
        self.t = np.zeros((self.nbc_u, self.nbc_u))
        self.p = np.zeros((self.nbc_u,))
        self.mu = np.zeros((self.nbc_x, 1 * len(data[0])))
        self.sigma = np.zeros((self.nbc_x, 1 * len(data[0]), 1 * len(data[0])))

        kmeans = KMeans(n_clusters=self.nbc_x).fit(data)
        hidden = kmeans.labels_
        hiddenc = np.stack((hidden[:-1], hidden[1:]), axis=-1)
        aux = np.moveaxis(np.indices((self.nbc_x, self.nbc_x)), 0, -1)
        broadc = (len(aux.shape) - len(hiddenc.shape) + 1)

        c = (1 / (len(data) - 1)) * (
            np.all(hiddenc.reshape((hiddenc.shape[0],) + (1,) * broadc + hiddenc.shape[1:]) == aux, axis=-1).sum(
                axis=0)).reshape((self.nbc_x, self.nbc_x))

        lxprime = np.copy(self.lx).T
        lxprime[np.invert(np.any(lxprime == 1, axis=1))] = 0
        u = lxprime @ (perturbation_param * c) @ lxprime.T
        u[u == 0] = (1 - perturbation_param) / (2 * u[u == 0].shape[0])
        u[-1, -1] = 0
        u[-1, -1] = 1 - u.sum()

        self.p = u.sum(axis=1)
        self.t = ((u).T / self.p).T

        self.mu = (((hidden[..., np.newaxis] == np.indices((self.mu.shape[0],)))[..., np.newaxis] * data[:, np.newaxis,
                                                                                                    ...]).sum(axis=0) /
                   (
                           hidden[..., np.newaxis] == np.indices((self.mu.shape[0],))).sum(axis=0)[
                       ..., np.newaxis]).reshape(self.mu.shape)
        self.sigma = (((hidden[..., np.newaxis] == np.indices((self.mu.shape[0],))).reshape(
            (hidden.shape[0], self.mu.shape[0]))[..., np.newaxis, np.newaxis] * np.einsum('...i,...j',
                                                                                          (data[:, np.newaxis, ...] -
                                                                                           self.mu[np.newaxis, ...]),
                                                                                          (data[:, np.newaxis, ...] -
                                                                                           self.mu[
                                                                                               np.newaxis, ...]))).sum(
            axis=0)
                      / ((hidden[..., np.newaxis] == np.indices((self.mu.shape[0],))).sum(
                    axis=0)).reshape((self.mu.shape[0],))[..., np.newaxis, np.newaxis])

    def init_from_markov_chain(self, data, iter=100, early_stopping=10 ** -4, perturbation_param=0.5, filepath=None):
        hmc = HMC_ctod(self.nbc_x)
        if filepath is None:
            hmc.init_kmeans(data)
            hmc.get_param_EM(data, iter, early_stopping=early_stopping)
        else:
            hmc.load_param_from_json(filepath)
        c = (hmc.t.T * hmc.p).T
        if self.lx is None:
            if self.nbc_u == (2 ** self.nbc_x) - 1:
                self.lx = np.stack([convertcls_vect(i + 1, (2,) * self.nbc_x) for i in range(self.nbc_u)], axis=0).T
                card = 1 / np.sum(self.lx, axis=0)
                self.lx=self.lx*card
            else:
                self.lx = np.vstack((np.eye(self.nbc_x), np.ones((self.nbc_x,)))).T
                card = 1 / np.sum(self.lx, axis=0)
                self.lx = self.lx * card

        lxprime = np.copy(self.lx).T
        lxprime[np.invert(np.any(lxprime == 1, axis=1))] = 0
        u = lxprime @ (perturbation_param * c) @ lxprime.T
        u[u == 0] = (1 - perturbation_param) / (2 * u[u == 0].shape[0])
        u[-1, -1] = 0
        u[-1, -1] = 1 - u.sum()

        self.p = u.sum(axis=1)
        self.t = ((u).T / self.p).T
        self.mu = hmc.mu
        self.sigma = hmc.sigma

    def give_param(self, u, mu, sigma):
        if self.lx is None:
            if self.nbc_u == (2 ** self.nbc_x) - 1:
                self.lx = np.stack([convertcls_vect(i + 1, (2,) * self.nbc_x) for i in range(self.nbc_u)], axis=0).T
                card = 1 / np.sum(self.lx, axis=0)
                self.lx=self.lx*card
            else:
                self.lx = np.vstack((np.eye(self.nbc_x), np.ones((self.nbc_x,)))).T
                card = 1 / np.sum(self.lx, axis=0)
                self.lx = self.lx * card
        self.p = u.sum(axis=1)
        self.t = ((u).T / self.p).T
        self.t[np.isnan(self.t)] = 0
        self.mu = mu
        self.sigma = sigma

    def get_param_give_form(self):
        u = (self.t.T * self.p).T
        return u, self.mu, self.sigma

    def get_param_apri_sup(self, i,j):
        P = (calc_matDS((self.t.T * self.p).T, self.lx)).sum(axis=1)
        T = calc_transDS(self.t, self.lx)
        if i is not None:
            p = np.zeros(P.shape)
            p[i * self.nbc_u: (i + 1) * self.nbc_u] = P[i * self.nbc_u: (i + 1) * self.nbc_u]
            if j is not None:
                t = np.zeros(T.shape)
                t[i * self.nbc_u: (i + 1) * self.nbc_u, j * self.nbc_u: (j + 1) * self.nbc_u] = T[i * self.nbc_u: (i + 1) * self.nbc_u, j * self.nbc_u: (j + 1) * self.nbc_u]
            else:
                t = np.zeros(T.shape)
                t[i * self.nbc_u: (i + 1) * self.nbc_u, :] = T[i * self.nbc_u: (i + 1) * self.nbc_u, :]
        elif j is not None:
            t = np.zeros(T.shape)
            t[:, j * self.nbc_u: (j + 1) * self.nbc_u] = T[:, j * self.nbc_u: (j + 1) * self.nbc_u]
            p = P
        else:
            p = P
            t = T
        return p, t

    def get_all_params_apri_sup(self, hidden):
        params = [self.get_param_apri_sup(hidden[i],hidden[i+1]) for i in range(hidden.shape[0]-1)]
        P = np.array([ele[0] for ele in params] + [self.get_param_apri_sup(hidden[-1],None)[0]])
        T = np.array([ele[1] for ele in params])
        return P,T

    def save_param_to_json(self, filepath):
        param_s = {'p': self.p.tolist(), 't': self.t.tolist(), 'mu': self.mu.tolist(),
                   'sig': self.sigma.tolist()}
        with open(filepath,'w') as f:
            json.dump(param_s, f, ensure_ascii=False)

    def load_param_from_json(self, filepath):
        with open(filepath, 'r') as f:
            params = json.load(f)
        if self.lx is None:
            if self.nbc_u == (2 ** self.nbc_x) - 1:
                self.lx = np.stack([convertcls_vect(i + 1, (2,) * self.nbc_x) for i in range(self.nbc_u)], axis=0).T
                card = 1 / np.sum(self.lx, axis=0)
                self.lx=self.lx*card
            else:
                self.lx = np.vstack((np.eye(self.nbc_x), np.ones((self.nbc_x,)))).T
                card = 1 / np.sum(self.lx, axis=0)
                self.lx = self.lx * card
        self.p = np.array(params['p'])
        self.t = np.array(params['t'])
        self.mu = np.array(params['mu'])
        self.sigma = np.array(params['sig'])

    def seg_map(self, data):
        pass

    def seg_mpm(self, data, hidden=None):
        mu = np.repeat(self.mu, self.nbc_u, axis=0)
        sigma = np.repeat(self.sigma, self.nbc_u, axis=0)
        gaussians = np_multivariate_normal_pdf(data, mu, sigma)
        forward, backward = self.get_forward_backward(gaussians, hidden=hidden)
        if self.vect:
            p_apost = forward
            p_apost_x = (p_apost.reshape((p_apost.shape[0], self.nbc_x, self.nbc_u))).sum(axis=2)
            return np.argmax(p_apost_x, axis=1)
        else:
            res = np.zeros((data.shape[0]))
            for i in range(len(res)):
                p_apost_i = forward[i]
                p_apost_ix = (p_apost_i.reshape((self.nbc_x, self.nbc_u))).sum(axis=1)
                res[i] = np.argmax(p_apost_ix)
            return res

    def seg_mpm_u(self, data):
        mu = np.repeat(self.mu, self.nbc_u, axis=0)
        sigma = np.repeat(self.sigma, self.nbc_u, axis=0)
        gaussians = np_multivariate_normal_pdf(data, mu, sigma)
        forward, backward = self.get_forward_backward(gaussians)
        if self.vect:
            p_apost = forward
            p_apost_u = (p_apost.reshape((p_apost.shape[0], self.nbc_x, self.nbc_u))).sum(axis=1)
            return np.argmax(p_apost_u, axis=1)
        else:
            res = np.zeros((data.shape[0]))
            for i in range(len(res)):
                p_apost_i = forward[i] * backward[i]
                p_apost_i = p_apost_i / p_apost_i.sum()
                p_apost_iu = (p_apost_i.reshape((self.nbc_x, self.nbc_u))).sum(axis=0)
                res[i] = np.argmax(p_apost_iu)
            return res

    def simul_hidden_apost(self, backward, gaussians, x_only=False, hidden=None):
        res = np.zeros(len(backward), dtype=int)
        T = calc_transDS(self.t, self.lx)
        T[np.isnan(T)] = 0
        aux = backward[0] / backward[0].sum()
        test = np.random.multinomial(1, aux)
        res[0] = np.argmax(test)
        if self.vect:
            T = T[np.newaxis, :, :]
            if hidden is not None:
                P, T = self.get_all_params_apri_sup(hidden)
            tapost = (
                (gaussians[1:, np.newaxis, :]
                 * backward[1:, np.newaxis, :]
                 * T)
            )
            tapost = tapost / tapost.sum(axis=2)[..., np.newaxis]
            tapost[np.isnan(tapost)] = 0
            for i in range(1, len(res)):
                test = np.random.multinomial(1, tapost[i - 1, res[i - 1], :])
                res[i] = np.argmax(test)
        else:
            for i in range(1, len(res)):
                if hidden is not None:
                    P, T = self.get_param_apri_sup(hidden[i-1], hidden[i])
                tapost_i = (
                    (gaussians[i, np.newaxis, :]
                     * backward[i, np.newaxis, :]
                     * T[:, :])
                )
                tapost_i = tapost_i / tapost_i.sum(axis=1)[..., np.newaxis]
                tapost_i[np.isnan(tapost_i)] = 0
                test = np.random.multinomial(1, tapost_i[res[i - 1], :])
                res[i] = np.argmax(test)
        if x_only:
            res = convert_multcls_vectors(res, (self.nbc_u, self.nbc_x))[:, 1]
        return res

    def simul_visible(self, hidden):
        res = np.zeros((len(hidden), self.mu.shape[-1]))
        mu = np.repeat(self.mu, self.nbc_u, axis=0)
        sigma = np.repeat(self.sigma, self.nbc_u, axis=0)
        for i in range(0, len(hidden)):
            res[i] = multivariate_normal.rvs(mu[hidden[i]], sigma[hidden[i]])
        return np.array(res)

    def generate_sample(self, length, x_only=True):
        hidden = np.zeros(length, dtype=int)
        visible = np.zeros((length, self.mu.shape[-1]))
        backward = self.get_backward_apri(length)
        mu = np.repeat(self.mu, self.nbc_u, axis=0)
        sigma = np.repeat(self.sigma, self.nbc_u, axis=0)
        T = calc_transDS(self.t, self.lx)
        T[np.isnan(T)] = 0
        if self.vect:
            tapri = (
                (backward[1:, np.newaxis, :]
                 * T[np.newaxis, :, :])
            )
            tapri = tapri / tapri.sum(axis=2)[..., np.newaxis]
            tapri[np.isnan(tapri)] = 0
            test = np.random.multinomial(1, backward[0] / backward[0].sum())
            hidden[0] = np.argmax(test)
            visible[0] = multivariate_normal.rvs(mu[hidden[0]], sigma[hidden[0]])
            for i in range(1, length):
                test = np.random.multinomial(1, tapri[i - 1, hidden[i - 1], :])
                hidden[i] = np.argmax(test)
                visible[i] = multivariate_normal.rvs(mu[hidden[i]], sigma[hidden[i]])
        else:
            for i in range(1, length):
                tapri_i = (
                    (backward[i, np.newaxis, :]
                     * T[np.newaxis, :, :])
                )
                tapri_i = tapri_i / tapri_i.sum(axis=1)[..., np.newaxis]
                tapri_i[np.isnan(tapri_i)] = 0
                test = np.random.multinomial(1, tapri_i[hidden[i - 1], :])
                hidden[i] = np.argmax(test)
                visible[i] = multivariate_normal.rvs(mu[hidden[i]], sigma[hidden[i]])
        if x_only:
            hidden = convert_multcls_vectors(hidden, (self.nbc_u, self.nbc_x))[:, 1]
        return hidden, visible

    def get_backward_apri(self, length):
        C = calc_matDS((self.t.T * self.p).T, self.lx)
        T = calc_transDS(self.t, self.lx)
        T[np.isnan(T)] = 0
        backward = np.zeros((length, T.shape[0]))
        backward[length - 1] = np.ones(T.shape[0])
        for l in reversed(range(0, length - 1)):
            if l == 0:
                phi = C
            else:
                phi = T
            backward[l] = phi @ (backward[l + 1])
            backward[l] = backward[l] / (backward[l].sum())

        return backward

    def get_gaussians(self, data):
        mu = np.repeat(self.mu, self.nbc_u, axis=0)
        sigma = np.repeat(self.sigma, self.nbc_u, axis=0)
        return np_multivariate_normal_pdf(data, mu, sigma)

    def get_forward_backward(self, gaussians, log_forward=False, hidden=None):
        C = calc_matDS((self.t.T * self.p).T, self.lx)
        T = calc_transDS(self.t, self.lx)
        T[np.isnan(T)] = 0
        if hidden is not None:
            Pprime, Tprime = self.get_all_params_apri_sup(hidden)
        forward = np.zeros((len(gaussians), T.shape[0]))
        backward = np.zeros((len(gaussians), T.shape[0]))
        backward[len(gaussians) - 1] = np.ones(T.shape[0])
        if not log_forward:
            for l in reversed(range(0, len(gaussians) - 1)):
                if hidden is not None:
                    P, T = Pprime[l], Tprime[l]
                    C = (P * T.T).T
                if l == 0:
                    phi = ((C * gaussians[l + 1]).T * gaussians[l]).T
                else:
                    phi = T * gaussians[l + 1]
                backward[l] = phi @ (backward[l + 1])
                backward[l] = backward[l] / (backward[l].sum())

            forward[0] = backward[0] / np.sum(backward[0])
            if self.vect:
                T = T[np.newaxis, :, :]
                if hidden is not None:
                    T=Tprime
                tapost = (
                        (backward[1:, np.newaxis, :] * gaussians[1:, np.newaxis, :]
                         * T) / backward[:-1, :, np.newaxis])
                tapost[np.isnan(tapost)] = 0
                tapost = tapost / tapost.sum(axis=2)[..., np.newaxis]
                tapost[np.isnan(tapost)] = 0
                for k in range(1, len(gaussians)):
                    forward[k] = (forward[k - 1] @ tapost[k - 1])
            else:
                for k in range(1, len(gaussians)):
                    if hidden is not None:
                        P, T = Pprime[k-1], Tprime[k-1]
                    tapost_k = (
                        (gaussians[k, np.newaxis, :]
                         * backward[k, np.newaxis, :]
                         * T[:, :])
                    )
                    tapost_k = tapost_k / tapost_k.sum(axis=1)[..., np.newaxis]
                    tapost_k[np.isnan(tapost_k)] = 0
                    forward[k] = (forward[k - 1] @ tapost_k)

            return forward, backward
        else:
            log_forward = np.zeros((len(gaussians), T.shape[0]))
            log_forward[len(gaussians) - 1] = np.zeros(T.shape[0])
            for l in reversed(range(0, len(gaussians) - 1)):
                if hidden is not None:
                    P, T = Pprime[l], Tprime[l]
                    C=(P*T.T).T
                if l == 0:
                    phi = ((C * gaussians[l + 1]).T * gaussians[l]).T
                else:
                    phi = T * gaussians[l + 1]
                log_forward[l] = logsumexp((np.log(phi) + log_forward[l + 1]), axis=1)
                backward[l] = phi @ (backward[l + 1])
                backward[l] = backward[l] / (backward[l].sum())

            forward[0] = backward[0] / np.sum(backward[0])
            if self.vect:
                T = T[np.newaxis, :, :]
                if hidden is not None:
                    T=Tprime
                tapost = (
                        (backward[1:, np.newaxis, :] * gaussians[1:, np.newaxis, :]
                         * T) / backward[:-1, :, np.newaxis])
                tapost[np.isnan(tapost)] = 0
                tapost = tapost / tapost.sum(axis=2)[..., np.newaxis]
                tapost[np.isnan(tapost)] = 0
                for k in range(1, len(gaussians)):
                    forward[k] = (forward[k - 1] @ tapost[k - 1])
            else:
                for k in range(1, len(gaussians)):
                    if hidden is not None:
                        P, T = Pprime[k-1], Tprime[k-1]
                    tapost_k = (
                        (gaussians[k, np.newaxis, :]
                         * backward[k, np.newaxis, :]
                         * T[:, :])
                    )
                    tapost_k = tapost_k / tapost_k.sum(axis=1)[..., np.newaxis]
                    tapost_k[np.isnan(tapost_k)] = 0
                    forward[k] = (forward[k - 1] @ tapost_k)
            return forward, backward, log_forward

    def calc_param_EM(self, data, forward, backward, gaussians, hidden=None):
        T = calc_transDS(self.t, self.lx)
        if self.vect:
            T = T[np.newaxis, :, :]
            if hidden is not None:
                P, T = self.get_all_params_apri_sup(hidden)
            tapost = (
                (backward[1:, np.newaxis, :] * gaussians[1:, np.newaxis, :]
                 * T))
            tapost = tapost / tapost.sum(axis=2)[..., np.newaxis]
            tapost[np.isnan(tapost)] = 0
            gamma = tapost * forward[:-1, :, np.newaxis]
            psi = forward
            aux = (psi[:-1].reshape(((psi.shape[0]-1), self.nbc_x, self.nbc_u))).sum(
                axis=(0, 1))
            aux[aux == 0] = 1 * 10 ** -100
            self.p = (1 / psi.shape[0]) * (psi.reshape((psi.shape[0], self.nbc_x, self.nbc_u))).sum(
                axis=(0, 1))
            self.t = np.transpose(np.transpose((
                                                gamma.reshape((gamma.shape[0], self.nbc_x, self.nbc_u, self.nbc_x,
                                                               self.nbc_u)).sum(
                                                    axis=(0, 1, 3)))) / (aux))

            psi = (psi.reshape((psi.shape[0], self.nbc_x, self.nbc_u))).sum(axis=2)
            self.mu = (((psi[..., np.newaxis] * data[:, np.newaxis, ...]).sum(axis=0)) / (
                psi.sum(axis=0)[..., np.newaxis])).reshape(self.mu.shape)

            self.sigma = (psi.reshape((psi.shape[0], self.mu.shape[0]))[..., np.newaxis, np.newaxis] * np.einsum(
                '...i,...j',
                (data[:, np.newaxis, ...] - self.mu[np.newaxis, ...]),
                (data[:, np.newaxis, ...] -
                 self.mu[np.newaxis, ...])
            )).sum(
                axis=0) / (psi.sum(axis=0)[..., np.newaxis, np.newaxis])
        else:
            psi = forward[0]
            gamma = np.zeros(T.shape)
            aux = np.zeros(T.shape[0])
            mu_aux = (psi.reshape((self.nbc_x, self.nbc_u)).sum(axis=1)[..., np.newaxis] * data[0, np.newaxis, ...])
            sigma_aux = np.zeros(self.sigma.shape)
            for i in range(1, len(gaussians)):
                if hidden is not None:
                    P, T = self.get_param_apri_sup(hidden[i-1], hidden[i])
                aux = aux + forward[i - 1]
                psi = psi + forward[i]
                tapost_i = (
                    (gaussians[i, np.newaxis, :]
                     * backward[i, np.newaxis, :]
                     * T[:, :])
                )
                tapost_i = tapost_i / tapost_i.sum(axis=1)[..., np.newaxis]
                tapost_i[np.isnan(tapost_i)] = 0
                gamma = gamma + tapost_i * forward[i - 1, :, np.newaxis]
                mu_aux = mu_aux + ((forward[i]).reshape(
                    (self.nbc_x, self.nbc_u)).sum(axis=1)[..., np.newaxis] * data[i, np.newaxis, ...])
            self.t = np.transpose(np.transpose(
                gamma.reshape((self.nbc_x, self.nbc_u, self.nbc_x, self.nbc_u)).sum(axis=(1, 3))) / aux.reshape(
                (self.nbc_x, self.nbc_u)).sum(axis=0))
            self.p = (psi.reshape((self.nbc_x, self.nbc_u))).sum(axis=(0)) / len(data)
            self.mu = (mu_aux / psi.reshape((self.nbc_x, self.nbc_u)).sum(axis=1)[..., np.newaxis]).reshape(
                self.mu.shape)
            for i in range(len(gaussians)):
                sigma_aux = sigma_aux + ((forward[i]).reshape(
                    (self.nbc_x, self.nbc_u)).sum(axis=1)[
                                             ..., np.newaxis, np.newaxis] * np.einsum('...i,...j',
                                                                                      (data[i, np.newaxis, ...] -
                                                                                       self.mu[np.newaxis, ...]),
                                                                                      (data[i, np.newaxis, ...] -
                                                                                       self.mu[np.newaxis, ...])
                                                                                      )).reshape(self.sigma.shape)
            self.sigma = (
                    sigma_aux / psi.reshape((self.nbc_x, self.nbc_u)).sum(axis=1)[..., np.newaxis, np.newaxis])
        self.sigma = self.sigma+np.full(self.sigma.shape, np.eye(self.sigma.shape[-1]) * self.reg)

    def calc_param_ICE(self, data, forward, backward, gaussians, Nb_simul, hidden=None):
        s_hidden = np.stack([self.simul_hidden_apost(backward, gaussians) for n in range(Nb_simul)], axis=0)
        T = calc_transDS(self.t, self.lx)
        T[np.isnan(T)] = 0
        if self.vect:
            T = T[np.newaxis, :, :]
            if hidden is not None:
                P, T = self.get_all_params_apri_sup(hidden)
            tapost = (
                    (backward[1:, np.newaxis, :] * gaussians[1:, np.newaxis, :]
                     * T[np.newaxis, :, :]) / backward[:-1, :, np.newaxis])
            tapost[np.isnan(tapost)] = 0
            tapost = tapost / tapost.sum(axis=2)[..., np.newaxis]
            tapost[np.isnan(tapost)] = 0
            gamma = tapost * forward[:-1, :, np.newaxis]
            psi = forward
            aux = (psi[:-1].reshape(((psi.shape[0]-1), self.nbc_x, self.nbc_u))).sum(
                axis=(0, 1))
            aux[aux == 0] = 1 * 10 ** -100
            self.p = (1 / psi.shape[0]) * (psi.reshape((psi.shape[0], self.nbc_x, self.nbc_u))).sum(
                axis=(0, 1))
            self.t = np.transpose(np.transpose((
                                                gamma.reshape((gamma.shape[0], self.nbc_x, self.nbc_u, self.nbc_x,
                                                               self.nbc_u)).sum(
                                                    axis=(0, 1, 3)))) / (aux))

            self.mu = (
                    (((s_hidden[..., np.newaxis] // self.nbc_u) == np.indices((self.mu.shape[0],)))[
                         ..., np.newaxis] * data[:, np.newaxis, ...]).sum(
                        axis=(0, 1))
                    /
                    ((s_hidden[..., np.newaxis] // self.nbc_u) == np.indices((self.mu.shape[0],))).sum(
                        axis=(0, 1))[..., np.newaxis]).reshape(self.mu.shape)
            self.sigma = (
                    (((s_hidden[..., np.newaxis] // self.nbc_u) == np.indices((self.mu.shape[0],))).reshape(
                        (s_hidden.shape[0], s_hidden.shape[1], self.mu.shape[0]))[
                         ..., np.newaxis, np.newaxis] * np.einsum(
                        '...i,...j',
                        (data[:, np.newaxis, ...] -
                         self.mu[np.newaxis, ...]),
                        (data[:, np.newaxis, ...] -
                         self.mu[
                             np.newaxis, ...]))).sum(
                        axis=(0, 1)) /
                    (((s_hidden[..., np.newaxis] // self.nbc_u) == np.indices((self.mu.shape[0],))).sum(
                        axis=(0, 1))[..., np.newaxis, np.newaxis]).reshape(self.sigma.shape))
        else:
            psi = forward[0]
            psi_prime = (hidden[:, 0, np.newaxis] == np.indices((self.mu.shape[0],))).sum(axis=0)
            gamma = np.zeros(T.shape)
            aux = np.zeros(T.shape[0])
            mu_aux = (psi.reshape((self.nbc_x, self.nbc_u)).sum(axis=1) * data[0])
            sigma_aux = np.zeros(self.sigma.shape)
            for i in range(1, len(gaussians)):
                aux = aux + forward[i - 1]
                psi = psi + forward[i]
                if hidden is not None:
                    P, T = self.get_param_apri_sup(hidden[i-1], hidden[i])
                tapost_i = (
                    (gaussians[i, np.newaxis, :]
                     * backward[i, np.newaxis, :]
                     * T[:, :])
                )
                tapost_i = tapost_i / tapost_i.sum(axis=1)[..., np.newaxis]
                tapost_i[np.isnan(tapost_i)] = 0
                gamma = gamma + tapost_i * forward[i - 1, :, np.newaxis]
                psi_prime = psi_prime + (
                        (s_hidden[:, i, np.newaxis] // self.nbc_u) == np.indices((self.mu.shape[0],))).sum(axis=0)
                mu_aux = mu_aux + (
                        ((s_hidden[:, i, np.newaxis] // self.nbc_u) == np.indices((self.mu.shape[0],))) * data[
                    i]).sum(
                    axis=0)
            self.t = np.transpose(np.transpose(
                gamma.reshape((self.nbc_x, self.nbc_u, self.nbc_x, self.nbc_u)).sum(axis=(1, 3))) / aux.reshape(
                (self.nbc_x, self.nbc_u)).sum(axis=0))
            self.p = (psi.reshape((self.nbc_x, self.nbc_u))).sum(axis=(0)) / len(data)
            self.mu = (mu_aux / psi_prime).reshape(self.mu.shape)
            for i in range(len(gaussians)):
                sigma_aux = sigma_aux + (
                        ((s_hidden[:, i, np.newaxis] // self.nbc_u) == np.indices((self.mu.shape[0],)))[
                            ..., np.newaxis, np.newaxis] * np.einsum('...i,...j',
                                                                     (data[i, np.newaxis, ...] -
                                                                      self.mu[np.newaxis, ...]),
                                                                     (data[i, np.newaxis, ...] -
                                                                      self.mu[np.newaxis, ...])
                                                                     )).sum(axis=0).reshape(
                    self.sigma.shape)
            self.sigma = (sigma_aux / psi_prime[..., np.newaxis, np.newaxis])
        self.sigma = self.sigma+np.full(self.sigma.shape, np.eye(self.sigma.shape[-1]) * self.reg)

    def calc_param_SEM(self, data, forward, backward, gaussians, hidden=None):
        T = calc_transDS(self.t, self.lx)
        hidden = self.simul_hidden_apost(backward, gaussians, hidden=hidden)
        hiddenc = np.stack((hidden[:-1], hidden[1:]), axis=-1)
        aux = np.moveaxis(np.indices(T.shape), 0, -1)
        if self.vect:
            c = (1 / hiddenc.shape[0]) * (
                np.all(hiddenc[:, np.newaxis, np.newaxis, ...] == aux[np.newaxis, ...], axis=-1).reshape(
                    (hiddenc.shape[0], self.nbc_x, self.nbc_u, self.nbc_x, self.nbc_u)).sum(
                    axis=(0, 1, 3)))
            self.p = (1 / hidden.shape[0]) *(hidden[..., np.newaxis] == np.indices((T.shape[0],))).reshape(
                (hidden.shape[0], self.nbc_x, self.nbc_u)).sum(axis=(0, 1))
            self.t = (c.T / self.p).T
            self.t[np.isnan(self.t)] = 0
            self.mu = ((((hidden[..., np.newaxis] // self.nbc_u) == np.indices((self.mu.shape[0],)))[
                            ..., np.newaxis] * data[:,
                                               np.newaxis,
                                               ...]).sum(
                axis=(0)) / (
                               (hidden[..., np.newaxis] // self.nbc_u) == np.indices((self.mu.shape[0],))).sum(
                axis=(0))[
                           ..., np.newaxis]).reshape(self.mu.shape)
            self.sigma = ((((hidden[..., np.newaxis] // self.nbc_u) == np.indices((self.mu.shape[0],))).reshape(
                (hidden.shape[0], self.mu.shape[0]))[..., np.newaxis, np.newaxis] * np.einsum(
                '...i,...j',
                (data[:, np.newaxis,
                 ...] - self.mu[
                     np.newaxis, ...]),
                (data[:, np.newaxis,
                 ...] -
                 self.mu[
                     np.newaxis, ...]))).sum(
                axis=(0))
                          / (((hidden[..., np.newaxis] // self.nbc_u) == np.indices((self.mu.shape[0],))).sum(
                        axis=(0))).reshape((self.mu.shape[0],))[..., np.newaxis, np.newaxis])
        else:
            c = np.zeros(self.t.shape)
            p = ((hidden[0, np.newaxis] // self.nbc_u) == np.indices((T.shape[0],))).reshape(
                (1, self.nbc_x, self.nbc_u)).sum(axis=(0, 1))
            psi_prime = ((hidden[0, np.newaxis] // self.nbc_u) == np.indices((self.mu.shape[0],)))
            mu_aux = (((hidden[0, np.newaxis] // self.nbc_u) == np.indices((self.mu.shape[0],))) * data[0])
            sigma_aux = np.zeros(self.sigma.shape)
            for i in range(1, len(data)):
                c = c + (
                    np.all(hiddenc[i - 1, np.newaxis, np.newaxis, ...] == aux[np.newaxis, ...], axis=-1)).reshape(
                    (1, self.nbc_x, self.nbc_u, self.nbc_x, self.nbc_u)).sum(
                    axis=(0, 1, 3))
                p = p + (hidden[i, np.newaxis] == np.indices((T.shape[0],))).reshape(
                    (1, self.nbc_x, self.nbc_u)).sum(axis=(0, 1))
                psi_prime = psi_prime + (
                        (hidden[i, np.newaxis] // self.nbc_u) == np.indices((self.mu.shape[0],))).sum(axis=0)
                mu_aux = mu_aux + (
                        ((hidden[i, np.newaxis] // self.nbc_u) == np.indices((self.mu.shape[0],))) * data[
                    i]).sum(axis=0)
            self.p = (1 / (len(data))) * p
            c = (1 / (len(data) - 1)) * c
            self.t = (c.T / self.p).T
            self.mu = (mu_aux / psi_prime).reshape(self.mu.shape)
            for i in range(len(data)):
                sigma_aux = sigma_aux + (((hidden[i, np.newaxis] // self.nbc_u) == np.indices((self.mu.shape[0],)))[
                                             ..., np.newaxis, np.newaxis] * np.einsum('...i,...j',
                                                                                      (data[i, np.newaxis, ...] -
                                                                                       self.mu[np.newaxis, ...]),
                                                                                      (data[i, np.newaxis, ...] -
                                                                                       self.mu[np.newaxis, ...])
                                                                                      )).sum(axis=0).reshape(
                    self.sigma.shape)
            self.sigma = (sigma_aux / psi_prime[..., np.newaxis, np.newaxis]).reshape(self.sigma.shape)
        self.sigma = self.sigma+np.full(self.sigma.shape, np.eye(self.sigma.shape[-1]) * self.reg)

    def get_param_EM(self, data, iter, early_stopping=0, true_ll=False, hidden=None):
        print({'iter': 0, 'p': self.p, 't': self.t, 'mu': self.mu, 'sigma': self.sigma})
        prev_log_likelihood = 0
        for q in range(iter):
            mu = np.repeat(self.mu, self.nbc_u, axis=0)
            sigma = np.repeat(self.sigma, self.nbc_u, axis=0)
            gaussians = np_multivariate_normal_pdf(data, mu, sigma)
            if true_ll:
                forward, backward, log_forward = self.get_forward_backward(gaussians, log_forward=True, hidden=hidden)
                new_log_likelihood = logsumexp(log_forward[0])
            else:
                forward, backward = self.get_forward_backward(gaussians, hidden=hidden)
                prev_p = self.p
                prev_t = self.t
                prev_mu = self.mu
                prev_sigma = self.sigma
            self.calc_param_EM(data, forward, backward, gaussians, hidden=hidden)

            if early_stopping is not None:
                if true_ll:
                    diff_log_likelihood = np.abs((new_log_likelihood - prev_log_likelihood) / new_log_likelihood)
                    prev_log_likelihood = new_log_likelihood
                else:
                    diff_log_likelihood = np.sqrt(
                        ((self.p - prev_p) ** 2).sum() + ((self.t - prev_t) ** 2).sum() + (
                                (self.mu - prev_mu) ** 2).sum() + (
                                (self.sigma - prev_sigma) ** 2).sum())
                print({'iter': q + 1, 'p': self.p, 't': self.t, 'mu': self.mu, 'sigma': self.sigma,
                       'diff_norm_param': diff_log_likelihood})
                if diff_log_likelihood < early_stopping:
                    break
            else:
                print({'iter': q + 1, 'p': self.p, 't': self.t, 'mu': self.mu, 'sigma': self.sigma})

    def get_param_ICE(self, data, iter, Nb_simul, early_stopping=None, true_ll=False):
        print({'iter': 0, 'p': self.p, 't': self.t, 'mu': self.mu, 'sigma': self.sigma})
        prev_log_likelihood = 0
        for q in range(iter):
            mu = np.repeat(self.mu, self.nbc_u, axis=0)
            sigma = np.repeat(self.sigma, self.nbc_u, axis=0)
            gaussians = np_multivariate_normal_pdf(data, mu, sigma)
            if true_ll:
                forward, backward, log_forward = self.get_forward_backward(gaussians, log_forward=True)
                new_log_likelihood = logsumexp(log_forward[0])
            else:
                forward, backward = self.get_forward_backward(gaussians)
                prev_p = self.p
                prev_t = self.t
                prev_mu = self.mu
                prev_sigma = self.sigma
            self.calc_param_ICE(data, forward, backward, gaussians, Nb_simul)

            if early_stopping is not None:
                if true_ll:
                    diff_log_likelihood = np.abs((new_log_likelihood - prev_log_likelihood) / new_log_likelihood)
                    prev_log_likelihood = new_log_likelihood
                else:
                    diff_log_likelihood = np.sqrt(
                        ((self.p - prev_p) ** 2).sum() + ((self.t - prev_t) ** 2).sum() + (
                                (self.mu - prev_mu) ** 2).sum() + (
                                (self.sigma - prev_sigma) ** 2).sum())
                print({'iter': q + 1, 'p': self.p, 't': self.t, 'mu': self.mu, 'sigma': self.sigma,
                       'diff_norm_param': diff_log_likelihood})
                if diff_log_likelihood < early_stopping:
                    break
            else:
                print({'iter': q + 1, 'p': self.p, 't': self.t, 'mu': self.mu, 'sigma': self.sigma})

    def get_param_SEM(self, data, iter, early_stopping=None, true_ll=False):
        print({'iter': 0, 'p': self.p, 't': self.t, 'mu': self.mu, 'sigma': self.sigma})
        prev_log_likelihood = 0
        for q in range(iter):
            mu = np.repeat(self.mu, self.nbc_u, axis=0)
            sigma = np.repeat(self.sigma, self.nbc_u, axis=0)
            gaussians = np_multivariate_normal_pdf(data, mu, sigma)
            if true_ll:
                forward, backward, log_forward = self.get_forward_backward(gaussians, log_forward=True)
                new_log_likelihood = logsumexp(log_forward[0])
            else:
                forward, backward = self.get_forward_backward(gaussians)
                prev_p = self.p
                prev_t = self.t
                prev_mu = self.mu
                prev_sigma = self.sigma
            self.calc_param_SEM(data, forward, backward, gaussians)

            if early_stopping is not None:
                if true_ll:
                    diff_log_likelihood = np.abs((new_log_likelihood - prev_log_likelihood) / new_log_likelihood)
                    prev_log_likelihood = new_log_likelihood
                else:
                    diff_log_likelihood = np.sqrt(
                        ((self.p - prev_p) ** 2).sum() + ((self.t - prev_t) ** 2).sum() + (
                                (self.mu - prev_mu) ** 2).sum() + (
                                (self.sigma - prev_sigma) ** 2).sum())
                print({'iter': q + 1, 'p': self.p, 't': self.t, 'mu': self.mu, 'sigma': self.sigma,
                       'diff_norm_param': diff_log_likelihood})
                if diff_log_likelihood < early_stopping:
                    break
            else:
                print({'iter': q + 1, 'p': self.p, 't': self.t, 'mu': self.mu, 'sigma': self.sigma})


class HEMC_neigh_ctod:
    __slots__ = ('p', 't', 'lx', 'mu', 'sigma', 'nbc_x', 'nbc_u', 'vect', 'reg')

    def __init__(self, nbc_x, u='all', lx=None, p=None, t=None, mu=None, sigma=None, vect=True, reg=10**-10):
        self.reg = reg
        self.p = p
        self.t = t
        self.mu = mu
        self.sigma = sigma
        self.nbc_x = nbc_x
        if lx is None:
            if u == 'all':
                self.nbc_u = (2 ** nbc_x) - 1
            else:
                self.nbc_u = nbc_x + 1
        else:
            self.nbc_u = lx.shape[1]
        self.lx = lx
        self.vect = vect

    def init_data_prior(self, data, scale=1):
        if self.lx is None:
            if self.nbc_u == (2 ** self.nbc_x) - 1:
                self.lx = np.stack([convertcls_vect(i + 1, (2,) * self.nbc_x) for i in range(self.nbc_u)], axis=0).T
                card = 1 / np.sum(self.lx, axis=0)
                self.lx=self.lx*card
            else:
                self.lx = np.vstack((np.eye(self.nbc_x), np.ones((self.nbc_x,)))).T
                card = 1 / np.sum(self.lx, axis=0)
                self.lx = self.lx * card
        a = np.full((self.nbc_u, self.nbc_u), 1 / (2 * (self.nbc_u - 1)))
        a = a - np.diag(np.diag(a))
        p = np.array([1 / self.nbc_u] * self.nbc_u)
        t = np.diag(np.array([1 / 2] * self.nbc_u)) + a
        u = (t.T / p).T
        self.p = u.sum(axis=1)
        self.t = ((u).T / self.p).T

        M = np.mean(data, axis=0)
        Sig = np.cov(data, rowvar=False).reshape(data.shape[1], data.shape[1])
        self.mu = [None] * self.nbc_x
        self.sigma = [None] * self.nbc_x
        for l in range(self.nbc_x):
            if l % 2 == 0:
                self.mu[l] = M - scale * ((l / (self.nbc_x / 2)) * np.sum(Sig, axis=1))
            else:
                self.mu[l] = M + scale * ((l / (self.nbc_x / 2)) * np.sum(Sig, axis=1))
            self.sigma[l] = Sig

        self.mu = np.array(self.mu)
        self.sigma = np.array(self.sigma)

    def init_kmeans(self, data, perturbation_param=0.5):
        if self.lx is None:
            if self.nbc_u == (2 ** self.nbc_x) - 1:
                self.lx = np.stack([convertcls_vect(i + 1, (2,) * self.nbc_x) for i in range(self.nbc_u)], axis=0).T
                card = 1 / np.sum(self.lx, axis=0)
                self.lx=self.lx*card
            else:
                self.lx = np.vstack((np.eye(self.nbc_x), np.ones((self.nbc_x,)))).T
                card = 1 / np.sum(self.lx, axis=0)
                self.lx = self.lx * card
        self.t = np.zeros((self.nbc_u, self.nbc_u))
        self.p = np.zeros((self.nbc_u,))
        self.mu = np.zeros((self.nbc_x, 1 * len(data[0])))
        self.sigma = np.zeros((self.nbc_x, 1 * len(data[0]), 1 * len(data[0])))

        kmeans = KMeans(n_clusters=self.nbc_x).fit(data)
        hidden = kmeans.labels_
        hiddenc = np.stack((hidden[:-1], hidden[1:]), axis=-1)
        aux = np.moveaxis(np.indices((self.nbc_x, self.nbc_x)), 0, -1)
        broadc = (len(aux.shape) - len(hiddenc.shape) + 1)

        c = (1 / (len(data) - 1)) * (
            np.all(hiddenc.reshape((hiddenc.shape[0],) + (1,) * broadc + hiddenc.shape[1:]) == aux, axis=-1).sum(
                axis=0)).reshape((self.nbc_x, self.nbc_x))

        lxprime = np.copy(self.lx).T
        lxprime[np.invert(np.any(lxprime == 1, axis=1))] = 0
        u = lxprime @ (perturbation_param * c) @ lxprime.T
        u[u == 0] = (1 - perturbation_param) / (2 * u[u == 0].shape[0])
        u[-1, -1] = 0
        u[-1, -1] = 1 - u.sum()

        self.p = u.sum(axis=1)
        self.t = ((u).T / self.p).T

        self.mu = (((hidden[..., np.newaxis] == np.indices((self.mu.shape[0],)))[..., np.newaxis] * data[:, np.newaxis,
                                                                                                    ...]).sum(axis=0) /
                   (
                           hidden[..., np.newaxis] == np.indices((self.mu.shape[0],))).sum(axis=0)[
                       ..., np.newaxis]).reshape(self.mu.shape)
        self.sigma = (((hidden[..., np.newaxis] == np.indices((self.mu.shape[0],))).reshape(
            (hidden.shape[0], self.mu.shape[0]))[..., np.newaxis, np.newaxis] * np.einsum('...i,...j',
                                                                                          (data[:, np.newaxis, ...] -
                                                                                           self.mu[np.newaxis, ...]),
                                                                                          (data[:, np.newaxis, ...] -
                                                                                           self.mu[
                                                                                               np.newaxis, ...]))).sum(
            axis=0)
                      / ((hidden[..., np.newaxis] == np.indices((self.mu.shape[0],))).sum(
                    axis=0)).reshape((self.mu.shape[0],))[..., np.newaxis, np.newaxis])

    def init_from_markov_chain(self, data, iter=100, early_stopping=10 ** -4, perturbation_param=0.5, filepath=None):
        hmc = HMC_ctod(self.nbc_x)
        if filepath is None:
            hmc.init_kmeans(data)
            hmc.get_param_EM(data, iter, early_stopping=early_stopping)
        else:
            hmc.load_param_from_json(filepath)
        c = (hmc.t.T * hmc.p).T
        if self.lx is None:
            if self.nbc_u == (2 ** self.nbc_x) - 1:
                self.lx = np.stack([convertcls_vect(i + 1, (2,) * self.nbc_x) for i in range(self.nbc_u)], axis=0).T
                card = 1 / np.sum(self.lx, axis=0)
                self.lx=self.lx*card
            else:
                self.lx = np.vstack((np.eye(self.nbc_x), np.ones((self.nbc_x,)))).T
                card = 1 / np.sum(self.lx, axis=0)
                self.lx = self.lx * card

        lxprime = np.copy(self.lx).T
        lxprime[np.invert(np.any(lxprime == 1, axis=1))] = 0
        u = lxprime @ (perturbation_param * c) @ lxprime.T
        u[u == 0] = (1 - perturbation_param) / (2 * u[u == 0].shape[0])
        u[-1, -1] = 0
        u[-1, -1] = 1 - u.sum()

        self.p = u.sum(axis=1)
        self.t = ((u).T / self.p).T
        self.mu = hmc.mu
        self.sigma = hmc.sigma

    def give_param(self, u, mu, sigma):
        if self.lx is None:
            if self.nbc_u == (2 ** self.nbc_x) - 1:
                self.lx = np.stack([convertcls_vect(i + 1, (2,) * self.nbc_x) for i in range(self.nbc_u)], axis=0).T
                card = 1 / np.sum(self.lx, axis=0)
                self.lx=self.lx*card
            else:
                self.lx = np.vstack((np.eye(self.nbc_x), np.ones((self.nbc_x,)))).T
                card = 1 / np.sum(self.lx, axis=0)
                self.lx = self.lx * card
        self.p = u.sum(axis=1)
        self.t = ((u).T / self.p).T
        self.t[np.isnan(self.t)] = 0
        self.mu = mu
        self.sigma = sigma

    def get_param_give_form(self):
        u = (self.t.T * self.p).T
        return u, self.mu, self.sigma

    def get_param_apri_sup(self, i,j):
        P = (calc_matDS((self.t.T * self.p).T, self.lx)).sum(axis=1)
        T = calc_transDS(self.t, self.lx)
        if i is not None:
            p = np.zeros(P.shape)
            p[i * self.nbc_u: (i + 1) * self.nbc_u] = P[i * self.nbc_u: (i + 1) * self.nbc_u]
            if j is not None:
                t = np.zeros(T.shape)
                t[i * self.nbc_u: (i + 1) * self.nbc_u, j * self.nbc_u: (j + 1) * self.nbc_u] = T[i * self.nbc_u: (i + 1) * self.nbc_u, j * self.nbc_u: (j + 1) * self.nbc_u]
            else:
                t = np.zeros(T.shape)
                t[i * self.nbc_u: (i + 1) * self.nbc_u, :] = T[i * self.nbc_u: (i + 1) * self.nbc_u, :]
        elif j is not None:
            t = np.zeros(T.shape)
            t[:, j * self.nbc_u: (j + 1) * self.nbc_u] = T[:, j * self.nbc_u: (j + 1) * self.nbc_u]
            p = P
        else:
            p = P
            t = T
        return p, t

    def get_all_params_apri_sup(self, hidden):
        params = [self.get_param_apri_sup(hidden[i],hidden[i+1]) for i in range(hidden.shape[0]-1)]
        P = np.array([ele[0] for ele in params] + [self.get_param_apri_sup(hidden[-1],None)[0]])
        T = np.array([ele[1] for ele in params])
        return P,T

    def get_gaussians(self, data, data_neighh, data_neighv):
        mu = np.repeat(self.mu, self.nbc_u, axis=0)
        sigma = np.repeat(self.sigma, self.nbc_u, axis=0)
        T = calc_transDS(self.t, self.lx)
        mask_data = np.isnan(data)
        mask_neighh = np.isnan(data_neighh)
        mask_neighv = np.isnan(data_neighv)
        data[mask_data] = 0
        data_neighh[mask_neighh] = 0
        data_neighv[mask_neighv] = 0
        neighh = np.einsum('...ij,...j', T, np_multivariate_normal_pdf(data_neighh, mu, sigma))
        neighv = np.einsum('...ij,...j', T, np_multivariate_normal_pdf(data_neighv, mu, sigma))
        gausses = np_multivariate_normal_pdf(data, mu, sigma)
        neighh[np.any(mask_neighh, axis=1)] = 1
        neighv[np.any(mask_neighv, axis=1)] = 1
        gausses[np.any(mask_data, axis=1)] = 1
        return gausses * neighh * neighv

    def save_param_to_json(self, filepath):
        param_s = {'p': self.p.tolist(), 't': self.t.tolist(), 'mu': self.mu.tolist(),
                   'sig': self.sigma.tolist()}
        with open(filepath,'w') as f:
            json.dump(param_s, f, ensure_ascii=False)

    def load_param_from_json(self, filepath):
        with open(filepath, 'r') as f:
            params = json.load(f)
        if self.lx is None:
            if self.nbc_u == (2 ** self.nbc_x) - 1:
                self.lx = np.stack([convertcls_vect(i + 1, (2,) * self.nbc_x) for i in range(self.nbc_u)], axis=0).T
                card = 1 / np.sum(self.lx, axis=0)
                self.lx=self.lx*card
            else:
                self.lx = np.vstack((np.eye(self.nbc_x), np.ones((self.nbc_x,)))).T
                card = 1 / np.sum(self.lx, axis=0)
                self.lx = self.lx * card
        self.p = np.array(params['p'])
        self.t = np.array(params['t'])
        self.mu = np.array(params['mu'])
        self.sigma = np.array(params['sig'])

    def seg_map(self, data):
        pass

    def seg_mpm(self, data, data_neighh, data_neighv, hidden=None):
        gaussians = self.get_gaussians(data, data_neighh, data_neighv)
        forward, backward = self.get_forward_backward(gaussians, hidden=None)
        if self.vect:
            p_apost = forward
            p_apost_x = (p_apost.reshape((p_apost.shape[0], self.nbc_x, self.nbc_u))).sum(axis=2)
            return np.argmax(p_apost_x, axis=1)
        else:
            res = np.zeros((data.shape[0]))
            for i in range(len(res)):
                p_apost_i = forward[i]
                p_apost_ix = (p_apost_i.reshape((self.nbc_x, self.nbc_u))).sum(axis=1)
                res[i] = np.argmax(p_apost_ix)
            return res

    def seg_mpm_u(self, data, data_neighh, data_neighv):
        gaussians = self.get_gaussians(data, data_neighh, data_neighv)
        forward, backward = self.get_forward_backward(gaussians)
        if self.vect:
            p_apost = forward
            p_apost_u = (p_apost.reshape((p_apost.shape[0], self.nbc_x, self.nbc_u))).sum(axis=1)
            return np.argmax(p_apost_u, axis=1)
        else:
            res = np.zeros((data.shape[0]))
            for i in range(len(res)):
                p_apost_i = forward[i] * backward[i]
                p_apost_i = p_apost_i / p_apost_i.sum()
                p_apost_iu = (p_apost_i.reshape((self.nbc_x, self.nbc_u))).sum(axis=0)
                res[i] = np.argmax(p_apost_iu)
            return res

    def simul_hidden_apost(self, backward, gaussians, x_only=False, hidden=None):
        res = np.zeros(len(backward), dtype=int)
        T = calc_transDS(self.t, self.lx)
        T[np.isnan(T)] = 0
        aux = backward[0] / backward[0].sum()
        test = np.random.multinomial(1, aux)
        res[0] = np.argmax(test)
        if self.vect:
            T = T[np.newaxis, :, :]
            if hidden is not None:
                P, T = self.get_all_params_apri_sup(hidden)
            tapost = (
                (gaussians[1:, np.newaxis, :]
                 * backward[1:, np.newaxis, :]
                 * T)
            )
            tapost = tapost / tapost.sum(axis=2)[..., np.newaxis]
            tapost[np.isnan(tapost)] = 0
            for i in range(1, len(res)):
                test = np.random.multinomial(1, tapost[i - 1, res[i - 1], :])
                res[i] = np.argmax(test)
        else:
            for i in range(1, len(res)):
                if hidden is not None:
                    P, T = self.get_param_apri_sup(hidden[i-1], hidden[i])
                tapost_i = (
                    (gaussians[i, np.newaxis, :]
                     * backward[i, np.newaxis, :]
                     * T[:, :])
                )
                tapost_i = tapost_i / tapost_i.sum(axis=1)[..., np.newaxis]
                tapost_i[np.isnan(tapost_i)] = 0
                test = np.random.multinomial(1, tapost_i[res[i - 1], :])
                res[i] = np.argmax(test)
        if x_only:
            res = convert_multcls_vectors(res, (self.nbc_u, self.nbc_x))[:, 1]
        return res

    def simul_visible(self, hidden):
        res = np.zeros((len(hidden), self.mu.shape[-1]))
        mu = np.repeat(self.mu, self.nbc_u, axis=0)
        sigma = np.repeat(self.sigma, self.nbc_u, axis=0)
        for i in range(0, len(hidden)):
            res[i] = multivariate_normal.rvs(mu[hidden[i]], sigma[hidden[i]])
        return np.array(res)

    def generate_sample(self, length, x_only=True):
        hidden = np.zeros(length, dtype=int)
        visible = np.zeros((length, self.mu.shape[-1]))
        backward = self.get_backward_apri(length)
        mu = np.repeat(self.mu, self.nbc_u, axis=0)
        sigma = np.repeat(self.sigma, self.nbc_u, axis=0)
        T = calc_transDS(self.t, self.lx)
        T[np.isnan(T)] = 0
        if self.vect:
            tapri = (
                (backward[1:, np.newaxis, :]
                 * T[np.newaxis, :, :])
            )
            tapri = tapri / tapri.sum(axis=2)[..., np.newaxis]
            tapri[np.isnan(tapri)] = 0
            test = np.random.multinomial(1, backward[0] / backward[0].sum())
            hidden[0] = np.argmax(test)
            visible[0] = multivariate_normal.rvs(mu[hidden[0]], sigma[hidden[0]])
            for i in range(1, length):
                test = np.random.multinomial(1, tapri[i - 1, hidden[i - 1], :])
                hidden[i] = np.argmax(test)
                visible[i] = multivariate_normal.rvs(mu[hidden[i]], sigma[hidden[i]])
        else:
            for i in range(1, length):
                tapri_i = (
                    (backward[i, np.newaxis, :]
                     * T[np.newaxis, :, :])
                )
                tapri_i = tapri_i / tapri_i.sum(axis=1)[..., np.newaxis]
                tapri_i[np.isnan(tapri_i)] = 0
                test = np.random.multinomial(1, tapri_i[hidden[i - 1], :])
                hidden[i] = np.argmax(test)
                visible[i] = multivariate_normal.rvs(mu[hidden[i]], sigma[hidden[i]])
        if x_only:
            hidden = convert_multcls_vectors(hidden, (self.nbc_u, self.nbc_x))[:, 1]
        return hidden, visible

    def get_backward_apri(self, length):
        C = calc_matDS((self.t.T * self.p).T, self.lx)
        T = calc_transDS(self.t, self.lx)
        T[np.isnan(T)] = 0
        backward = np.zeros((length, T.shape[0]))
        backward[length - 1] = np.ones(T.shape[0])
        for l in reversed(range(0, length - 1)):
            if l == 0:
                phi = C
            else:
                phi = T
            backward[l] = phi @ (backward[l + 1])
            backward[l] = backward[l] / (backward[l].sum())

        return backward

    def get_forward_backward(self, gaussians, log_forward=False, hidden=None):
        C = calc_matDS((self.t.T * self.p).T, self.lx)
        T = calc_transDS(self.t, self.lx)
        T[np.isnan(T)] = 0
        if hidden is not None:
            Pprime, Tprime = self.get_all_params_apri_sup(hidden)
        forward = np.zeros((len(gaussians), T.shape[0]))
        backward = np.zeros((len(gaussians), T.shape[0]))
        backward[len(gaussians) - 1] = np.ones(T.shape[0])
        if not log_forward:
            for l in reversed(range(0, len(gaussians) - 1)):
                if hidden is not None:
                    P, T = Pprime[l], Tprime[l]
                    C = (P * T.T).T
                if l == 0:
                    phi = ((C * gaussians[l + 1]).T * gaussians[l]).T
                else:
                    phi = T * gaussians[l + 1]
                backward[l] = phi @ (backward[l + 1])
                backward[l] = backward[l] / (backward[l].sum())

            forward[0] = backward[0] / np.sum(backward[0])
            if self.vect:
                T = T[np.newaxis, :, :]
                if hidden is not None:
                    T=Tprime
                tapost = (
                        (backward[1:, np.newaxis, :] * gaussians[1:, np.newaxis, :]
                         * T) / backward[:-1, :, np.newaxis])
                tapost[np.isnan(tapost)] = 0
                tapost = tapost / tapost.sum(axis=2)[..., np.newaxis]
                tapost[np.isnan(tapost)] = 0
                for k in range(1, len(gaussians)):
                    forward[k] = (forward[k - 1] @ tapost[k - 1])
            else:
                for k in range(1, len(gaussians)):
                    if hidden is not None:
                        P, T = Pprime[k-1], Tprime[k-1]
                    tapost_k = (
                        (gaussians[k, np.newaxis, :]
                         * backward[k, np.newaxis, :]
                         * T[:, :])
                    )
                    tapost_k = tapost_k / tapost_k.sum(axis=1)[..., np.newaxis]
                    tapost_k[np.isnan(tapost_k)] = 0
                    forward[k] = (forward[k - 1] @ tapost_k)
            return forward, backward
        else:
            log_forward = np.zeros((len(gaussians), T.shape[0]))
            log_forward[len(gaussians) - 1] = np.zeros(T.shape[0])
            for l in reversed(range(0, len(gaussians) - 1)):
                if hidden is not None:
                    P, T = Pprime[l], Tprime[l]
                    C=(P*T.T).T
                if l == 0:
                    phi = ((C * gaussians[l + 1]).T * gaussians[l]).T
                else:
                    phi = T * gaussians[l + 1]
                log_forward[l] = logsumexp((np.log(phi) + log_forward[l + 1]), axis=1)
                backward[l] = phi @ (backward[l + 1])
                backward[l] = backward[l] / (backward[l].sum())

            forward[0] = backward[0] / np.sum(backward[0])
            if self.vect:
                T = T[np.newaxis, :, :]
                if hidden is not None:
                    T=Tprime
                tapost = (
                        (backward[1:, np.newaxis, :] * gaussians[1:, np.newaxis, :]
                         * T) / backward[:-1, :, np.newaxis])
                tapost[np.isnan(tapost)] = 0
                tapost = tapost / tapost.sum(axis=2)[..., np.newaxis]
                tapost[np.isnan(tapost)] = 0
                for k in range(1, len(gaussians)):
                    forward[k] = (forward[k - 1] @ tapost[k - 1])
            else:
                for k in range(1, len(gaussians)):
                    if hidden is not None:
                        P, T = Pprime[k-1], Tprime[k-1]
                    tapost_k = (
                        (gaussians[k, np.newaxis, :]
                         * backward[k, np.newaxis, :]
                         * T[:, :])
                    )
                    tapost_k = tapost_k / tapost_k.sum(axis=1)[..., np.newaxis]
                    tapost_k[np.isnan(tapost_k)] = 0
                    forward[k] = (forward[k - 1] @ tapost_k)
            return forward, backward, log_forward

    def calc_param_EM(self, data, forward, backward, gaussians, hidden=None):
        T = calc_transDS(self.t, self.lx)
        if self.vect:
            T = T[np.newaxis, :, :]
            if hidden is not None:
                P, T = self.get_all_params_apri_sup(hidden)
            tapost = (
                (backward[1:, np.newaxis, :] * gaussians[1:, np.newaxis, :]
                 * T))
            tapost = tapost / tapost.sum(axis=2)[..., np.newaxis]
            tapost[np.isnan(tapost)] = 0
            gamma = tapost * forward[:-1, :, np.newaxis]
            psi = forward
            aux = (psi.reshape((psi.shape[0], self.nbc_x, self.nbc_u))).sum(
                axis=(0, 1))
            aux[aux == 0] = 1 * 10 ** -100
            self.p = (1 / psi.shape[0]) * (psi.reshape((psi.shape[0], self.nbc_x, self.nbc_u))).sum(
                axis=(0, 1))
            self.t = np.transpose(np.transpose((
                                                gamma.reshape((gamma.shape[0], self.nbc_x, self.nbc_u, self.nbc_x,
                                                               self.nbc_u)).sum(
                                                    axis=(0, 1, 3)))) / (aux))

            psi = (psi.reshape((psi.shape[0], self.nbc_x, self.nbc_u))).sum(axis=2)
            self.mu = (((psi[..., np.newaxis] * data[:, np.newaxis, ...]).sum(axis=0)) / (
                psi.sum(axis=0)[..., np.newaxis])).reshape(self.mu.shape)

            self.sigma = (psi.reshape((psi.shape[0], self.mu.shape[0]))[..., np.newaxis, np.newaxis] * np.einsum(
                '...i,...j',
                (data[:, np.newaxis, ...] - self.mu[np.newaxis, ...]),
                (data[:, np.newaxis, ...] -
                 self.mu[np.newaxis, ...])
            )).sum(
                axis=0) / (psi.sum(axis=0)[..., np.newaxis, np.newaxis])
        else:
            psi = forward[0]
            gamma = np.zeros(T.shape)
            aux = np.zeros(T.shape[0])
            mu_aux = (psi.reshape((self.nbc_x, self.nbc_u)).sum(axis=1)[..., np.newaxis] * data[0, np.newaxis, ...])
            sigma_aux = np.zeros(self.sigma.shape)
            for i in range(1, len(gaussians)):
                if hidden is not None:
                    P, T = self.get_param_apri_sup(hidden[i-1], hidden[i])
                aux = aux + forward[i - 1]
                psi = psi + forward[i]
                tapost_i = (
                    (gaussians[i, np.newaxis, :]
                     * backward[i, np.newaxis, :]
                     * T[:, :])
                )
                tapost_i = tapost_i / tapost_i.sum(axis=1)[..., np.newaxis]
                tapost_i[np.isnan(tapost_i)] = 0
                gamma = gamma + tapost_i * forward[i - 1, :, np.newaxis]
                mu_aux = mu_aux + ((forward[i]).reshape(
                    (self.nbc_x, self.nbc_u)).sum(axis=1)[..., np.newaxis] * data[i, np.newaxis, ...])
            self.t = np.transpose(
                np.transpose(gamma.reshape((self.nbc_x, self.nbc_u, self.nbc_x,
                                                                   self.nbc_u)).sum(
                    axis=(0, 2))) / (aux))
            self.p = (psi.reshape((self.nbc_x, self.nbc_u))).sum(axis=(0)) / len(data)
            self.mu = (mu_aux / psi.reshape((self.nbc_x, self.nbc_u)).sum(axis=1)[..., np.newaxis]).reshape(
                self.mu.shape)
            for i in range(len(gaussians)):
                sigma_aux = sigma_aux + ((forward[i]).reshape(
                    (self.nbc_x, self.nbc_u)).sum(axis=1)[
                                             ..., np.newaxis, np.newaxis] * np.einsum('...i,...j',
                                                                                      (data[i, np.newaxis, ...] -
                                                                                       self.mu[np.newaxis, ...]),
                                                                                      (data[i, np.newaxis, ...] -
                                                                                       self.mu[np.newaxis, ...])
                                                                                      )).reshape(self.sigma.shape)
            self.sigma = (
                    sigma_aux / psi.reshape((self.nbc_x, self.nbc_u)).sum(axis=1)[..., np.newaxis, np.newaxis])
        self.sigma = self.sigma+np.full(self.sigma.shape, np.eye(self.sigma.shape[-1]) * self.reg)

    def calc_param_ICE(self, data, forward, backward, gaussians, Nb_simul):
        hidden = np.stack([self.simul_hidden_apost(backward, gaussians) for n in range(Nb_simul)], axis=0)
        T = calc_transDS(self.t, self.lx)
        T[np.isnan(T)] = 0
        if self.vect:
            tapost = (
                    (backward[1:, np.newaxis, :] * gaussians[1:, np.newaxis, :]
                     * T[np.newaxis, :, :]) / backward[:-1, :, np.newaxis])
            tapost[np.isnan(tapost)] = 0
            tapost = tapost / tapost.sum(axis=2)[..., np.newaxis]
            tapost[np.isnan(tapost)] = 0
            gamma = tapost * forward[:-1, :, np.newaxis]
            psi = forward
            aux = (psi.reshape((psi.shape[0], self.nbc_x, self.nbc_u))).sum(
                axis=(0, 1))
            aux[aux == 0] = 1 * 10 ** -100
            self.p = (1 / psi.shape[0]) * (psi.reshape((psi.shape[0], self.nbc_x, self.nbc_u))).sum(
                axis=(0, 1))
            self.t = np.transpose(np.transpose((
                                                gamma.reshape((gamma.shape[0], self.nbc_x, self.nbc_u, self.nbc_x,
                                                               self.nbc_u)).sum(
                                                    axis=(0, 1, 3)))) / (aux))

            self.mu = (
                    (((hidden[..., np.newaxis] // self.nbc_u) == np.indices((self.mu.shape[0],)))[
                         ..., np.newaxis] * data[:, np.newaxis, ...]).sum(
                        axis=(0, 1))
                    /
                    ((hidden[..., np.newaxis] // self.nbc_u) == np.indices((self.mu.shape[0],))).sum(
                        axis=(0, 1))[..., np.newaxis]).reshape(self.mu.shape)
            self.sigma = (
                    (((hidden[..., np.newaxis] // self.nbc_u) == np.indices((self.mu.shape[0],))).reshape(
                        (hidden.shape[0], hidden.shape[1], self.mu.shape[0]))[
                         ..., np.newaxis, np.newaxis] * np.einsum(
                        '...i,...j',
                        (data[:, np.newaxis, ...] -
                         self.mu[np.newaxis, ...]),
                        (data[:, np.newaxis, ...] -
                         self.mu[
                             np.newaxis, ...]))).sum(
                        axis=(0, 1)) /
                    (((hidden[..., np.newaxis] // self.nbc_u) == np.indices((self.mu.shape[0],))).sum(
                        axis=(0, 1))[..., np.newaxis, np.newaxis]).reshape(self.sigma.shape))
        else:
            psi = forward[0]
            psi_prime = (hidden[:, 0, np.newaxis] == np.indices((self.mu.shape[0],))).sum(axis=0)
            gamma = np.zeros(T.shape)
            aux = np.zeros(T.shape[0])
            mu_aux = (psi.reshape((self.nbc_x, self.nbc_u)).sum(axis=1) * data[0])
            sigma_aux = np.zeros(self.sigma.shape)
            for i in range(1, len(gaussians)):
                aux = aux + forward[i - 1]
                psi = psi + forward[i]
                tapost_i = (
                    (gaussians[i, np.newaxis, :]
                     * backward[i, np.newaxis, :]
                     * T[:, :])
                )
                tapost_i = tapost_i / tapost_i.sum(axis=1)[..., np.newaxis]
                tapost_i[np.isnan(tapost_i)] = 0
                gamma = gamma + tapost_i * forward[i - 1, :, np.newaxis]
                psi_prime = psi_prime + (
                        (hidden[:, i, np.newaxis] // self.nbc_u) == np.indices((self.mu.shape[0],))).sum(axis=0)
                mu_aux = mu_aux + (
                        ((hidden[:, i, np.newaxis] // self.nbc_u) == np.indices((self.mu.shape[0],))) * data[
                    i]).sum(
                    axis=0)
            self.t = np.transpose(
                np.transpose(gamma.reshape((self.nbc_x, self.nbc_u, self.nbc_x,
                                                                   self.nbc_u)).sum(
                    axis=(0, 2))) / (aux))
            self.p = (psi.reshape((self.nbc_x, self.nbc_u))).sum(axis=(0)) / len(data)
            self.mu = (mu_aux / psi_prime).reshape(self.mu.shape)
            for i in range(len(gaussians)):
                sigma_aux = sigma_aux + (
                        ((hidden[:, i, np.newaxis] // self.nbc_u) == np.indices((self.mu.shape[0],)))[
                            ..., np.newaxis, np.newaxis] * np.einsum('...i,...j',
                                                                     (data[i, np.newaxis, ...] -
                                                                      self.mu[np.newaxis, ...]),
                                                                     (data[i, np.newaxis, ...] -
                                                                      self.mu[np.newaxis, ...])
                                                                     )).sum(axis=0).reshape(
                    self.sigma.shape)
            self.sigma = (sigma_aux / psi_prime[..., np.newaxis, np.newaxis])
        self.sigma = self.sigma+np.full(self.sigma.shape, np.eye(self.sigma.shape[-1]) * self.reg)

    def calc_param_SEM(self, data, forward, backward, gaussians):
        T = calc_transDS((self.t.T / self.p).T, self.lx)
        hidden = self.simul_hidden_apost(backward, gaussians)
        hiddenc = np.stack((hidden[:-1], hidden[1:]), axis=-1)
        aux = np.moveaxis(np.indices(T.shape), 0, -1)
        if self.vect:
            c = (1 / hiddenc.shape[0]) * (
                np.all(hiddenc[:, np.newaxis, np.newaxis, ...] == aux[np.newaxis, ...], axis=-1).reshape(
                    (hiddenc.shape[0], self.nbc_x, self.nbc_u, self.nbc_x, self.nbc_u)).sum(
                    axis=(0, 1, 3)))
            self.p = (1 / hidden.shape[0]) *(hidden[..., np.newaxis] == np.indices((T.shape[0],))).reshape(
                (hidden.shape[0], self.nbc_x, self.nbc_u)).sum(axis=(0, 1))
            self.t = (c.T / self.p).T
            self.t[np.isnan(self.t)] = 0
            self.mu = ((((hidden[..., np.newaxis] // self.nbc_u) == np.indices((self.mu.shape[0],)))[
                            ..., np.newaxis] * data[:,
                                               np.newaxis,
                                               ...]).sum(
                axis=(0)) / (
                               (hidden[..., np.newaxis] // self.nbc_u) == np.indices((self.mu.shape[0],))).sum(
                axis=(0))[
                           ..., np.newaxis]).reshape(self.mu.shape)
            self.sigma = ((((hidden[..., np.newaxis] // self.nbc_u) == np.indices((self.mu.shape[0],))).reshape(
                (hidden.shape[0], self.mu.shape[0]))[..., np.newaxis, np.newaxis] * np.einsum(
                '...i,...j',
                (data[:, np.newaxis,
                 ...] - self.mu[
                     np.newaxis, ...]),
                (data[:, np.newaxis,
                 ...] -
                 self.mu[
                     np.newaxis, ...]))).sum(
                axis=(0))
                          / (((hidden[..., np.newaxis] // self.nbc_u) == np.indices((self.mu.shape[0],))).sum(
                        axis=(0))).reshape((self.mu.shape[0],))[..., np.newaxis, np.newaxis])
        else:
            c = np.zeros(self.t.shape)
            p = ((hidden[0, np.newaxis] // self.nbc_u) == np.indices((T.shape[0],))).reshape(
                (1, self.nbc_x, self.nbc_u)).sum(axis=(0, 1))
            psi_prime = ((hidden[0, np.newaxis] // self.nbc_u) == np.indices((self.mu.shape[0],)))
            mu_aux = (((hidden[0, np.newaxis] // self.nbc_u) == np.indices((self.mu.shape[0],))) * data[0])
            sigma_aux = np.zeros(self.sigma.shape)
            for i in range(1, len(data)):
                c = c + (
                    np.all(hiddenc[i - 1, np.newaxis, np.newaxis, ...] == aux[np.newaxis, ...], axis=-1)).reshape(
                    (1, self.nbc_x, self.nbc_u, self.nbc_x, self.nbc_u)).sum(
                    axis=(0, 1, 3))
                p = p + (hidden[i, np.newaxis] == np.indices((T.shape[0],))).reshape(
                    (1, self.nbc_x, self.nbc_u)).sum(axis=(0, 1))
                psi_prime = psi_prime + (
                        (hidden[i, np.newaxis] // self.nbc_u) == np.indices((self.mu.shape[0],))).sum(axis=0)
                mu_aux = mu_aux + (
                        ((hidden[i, np.newaxis] // self.nbc_u) == np.indices((self.mu.shape[0],))) * data[
                    i]).sum(axis=0)
            self.p = (1 / (len(data))) * p
            c = (1 / (len(data) - 1)) * c
            self.t = (c.T / self.p).T
            self.mu = (mu_aux / psi_prime).reshape(self.mu.shape)
            for i in range(len(data)):
                sigma_aux = sigma_aux + (((hidden[i, np.newaxis] // self.nbc_u) == np.indices((self.mu.shape[0],)))[
                                             ..., np.newaxis, np.newaxis] * np.einsum('...i,...j',
                                                                                      (data[i, np.newaxis, ...] -
                                                                                       self.mu[np.newaxis, ...]),
                                                                                      (data[i, np.newaxis, ...] -
                                                                                       self.mu[np.newaxis, ...])
                                                                                      )).sum(axis=0).reshape(
                    self.sigma.shape)
            self.sigma = (sigma_aux / psi_prime[..., np.newaxis, np.newaxis]).reshape(self.sigma.shape)
        self.sigma = self.sigma+np.full(self.sigma.shape, np.eye(self.sigma.shape[-1]) * self.reg)

    def get_param_EM(self, data, data_neighh, data_neighv, iter, early_stopping=0, true_ll=False, hidden=None):
        print({'iter': 0, 'p': self.p, 't': self.t, 'mu': self.mu, 'sigma': self.sigma})
        prev_log_likelihood = 0
        for q in range(iter):
            gaussians = self.get_gaussians(data, data_neighh, data_neighv)
            if true_ll:
                forward, backward, log_forward = self.get_forward_backward(gaussians, log_forward=True, hidden=hidden)
                new_log_likelihood = logsumexp(log_forward[0])
            else:
                forward, backward = self.get_forward_backward(gaussians, hidden=hidden)
                prev_p = self.p
                prev_t = self.t
                prev_mu = self.mu
                prev_sigma = self.sigma
            self.calc_param_EM(data, forward, backward, gaussians, hidden=hidden)

            if early_stopping is not None:
                if true_ll:
                    diff_log_likelihood = np.abs((new_log_likelihood - prev_log_likelihood) / new_log_likelihood)
                    prev_log_likelihood = new_log_likelihood
                else:
                    diff_log_likelihood = np.sqrt(
                        ((self.p - prev_p) ** 2).sum() + ((self.t - prev_t) ** 2).sum() + (
                                (self.mu - prev_mu) ** 2).sum() + (
                                (self.sigma - prev_sigma) ** 2).sum())
                print({'iter': q + 1, 'p': self.p, 't': self.t, 'mu': self.mu, 'sigma': self.sigma,
                       'diff_norm_param': diff_log_likelihood})
                if diff_log_likelihood < early_stopping:
                    break
            else:
                print({'iter': q + 1, 'p': self.p, 't': self.t, 'mu': self.mu, 'sigma': self.sigma})

    def get_param_ICE(self, data, data_neighh, data_neighv, iter, Nb_simul, early_stopping=None, true_ll=False):
        print({'iter': 0, 'p': self.p, 't': self.t, 'mu': self.mu, 'sigma': self.sigma})
        prev_log_likelihood = 0
        for q in range(iter):
            gaussians = self.get_gaussians(data, data_neighh, data_neighv)
            if true_ll:
                forward, backward, log_forward = self.get_forward_backward(gaussians, log_forward=True)
                new_log_likelihood = logsumexp(log_forward[0])
            else:
                forward, backward = self.get_forward_backward(gaussians)
                prev_p = self.p
                prev_t = self.t
                prev_mu = self.mu
                prev_sigma = self.sigma
            self.calc_param_ICE(data, forward, backward, gaussians, Nb_simul)

            if early_stopping is not None:
                if true_ll:
                    diff_log_likelihood = np.abs((new_log_likelihood - prev_log_likelihood) / new_log_likelihood)
                    prev_log_likelihood = new_log_likelihood
                else:
                    diff_log_likelihood = np.sqrt(
                        ((self.p - prev_p) ** 2).sum() + ((self.t - prev_t) ** 2).sum() + (
                                (self.mu - prev_mu) ** 2).sum() + (
                                (self.sigma - prev_sigma) ** 2).sum())
                print({'iter': q + 1, 'p': self.p, 't': self.t, 'mu': self.mu, 'sigma': self.sigma,
                       'diff_norm_param': diff_log_likelihood})
                if diff_log_likelihood < early_stopping:
                    break
            else:
                print({'iter': q + 1, 'p': self.p, 't': self.t, 'mu': self.mu, 'sigma': self.sigma})

    def get_param_SEM(self, data, data_neighh, data_neighv, iter, early_stopping=None, true_ll=False):
        print({'iter': 0, 'p': self.p, 't': self.t, 'mu': self.mu, 'sigma': self.sigma})
        prev_log_likelihood = 0
        for q in range(iter):
            gaussians = self.get_gaussians(data, data_neighh, data_neighv)
            if true_ll:
                forward, backward, log_forward = self.get_forward_backward(gaussians, log_forward=True)
                new_log_likelihood = logsumexp(log_forward[0])
            else:
                forward, backward = self.get_forward_backward(gaussians)
                prev_p = self.p
                prev_t = self.t
                prev_mu = self.mu
                prev_sigma = self.sigma
            self.calc_param_SEM(data, forward, backward, gaussians)

            if early_stopping is not None:
                if true_ll:
                    diff_log_likelihood = np.abs((new_log_likelihood - prev_log_likelihood) / new_log_likelihood)
                    prev_log_likelihood = new_log_likelihood
                else:
                    diff_log_likelihood = np.sqrt(
                        ((self.p - prev_p) ** 2).sum() + ((self.t - prev_t) ** 2).sum() + (
                                (self.mu - prev_mu) ** 2).sum() + (
                                (self.sigma - prev_sigma) ** 2).sum())
                print({'iter': q + 1, 'p': self.p, 't': self.t, 'mu': self.mu, 'sigma': self.sigma,
                       'diff_norm_param': diff_log_likelihood})
                if diff_log_likelihood < early_stopping:
                    break
            else:
                print({'iter': q + 1, 'p': self.p, 't': self.t, 'mu': self.mu, 'sigma': self.sigma})
