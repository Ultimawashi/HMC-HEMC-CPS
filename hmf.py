import numpy as np
import json
from math import factorial
from utils import add_border, crop_border, np_multivariate_normal_pdf, convert_vect_multcls, convert_vect_multcls_no_order, random_multinomial_vect
from sklearn.cluster import KMeans

class HMF_ctod:
    __slots__ = ('alpha', 'proba', 'mu', 'sigma', 'nbc_x', 'neigh', 'order', 'vect', 'reg')

    def __init__(self, nbc_x, neigh=4, alpha=None, proba=None, mu=None, sigma=None, order=False, vect=True, reg=10**-10):
        assert (neigh == 4 or neigh == 8), 'please choose only between 4 and 8 neighbour'
        self.alpha = alpha
        self.proba = proba
        self.mu = mu
        self.sigma = sigma
        self.nbc_x = nbc_x
        self.neigh = neigh
        self.order = order
        self.vect = vect
        self.reg = reg

    def calc_proba_champs(self):
        self.proba = np.exp(self.alpha)/np.exp(self.alpha).sum(axis=1)[...,np.newaxis]

    def calc_proba_champs_apost(self, gauss, reg=10 ** -10):
        return self.proba * gauss[np.newaxis,...] / ((self.proba * gauss[np.newaxis,...]).sum(axis=-1) + reg)[
            ..., np.newaxis]

    def calc_alpha_from_proba(self):
        self.alpha = np.zeros(self.proba.shape)
        for i in range(0, self.alpha.shape[0]):
            self.alpha[i, 0] = np.log(self.proba[i,0])
            for j in range(1, self.alpha.shape[1]):
                self.alpha[i, j] = np.log(self.proba[i, j] / (self.proba[i,0]+self.reg)) + self.alpha[i, 0]

    def config_voisinage(self, X, i, j):
        idx_class = X[:, i, j]
        X = X[:,(i - 1):(i + 2), (j - 1):(j + 2)]
        if self.neigh==4:
            mask = np.array([[False, True, False], [True, False, True], [False, True, False]])
        elif self.neigh==8:
            mask = np.array([[True, True, True], [True, False, True], [True, True, True]])
        else:
            mask = np.array([[False, True, False], [True, False, True], [False, True, False]])
        if self.order:
            idx_conf = convert_vect_multcls(X[:,mask].reshape(X.shape[0],4), (self.nbc_x,self.nbc_x,self.nbc_x,self.nbc_x))
        else :
            idx_conf = convert_vect_multcls_no_order(X[:, mask].reshape(X.shape[0], 4), (self.nbc_x, self.nbc_x, self.nbc_x, self.nbc_x))
        return idx_conf, idx_class

    def get_gaussians(self, data):
        gausses = np_multivariate_normal_pdf(data, self.mu, self.sigma)
        return gausses

    def estim_proba_apri(self, X):
        if self.order:
            p_apri = np.zeros((self.neigh**self.nbc_x,self.nbc_x))
        else:
            nbconf = int(factorial(self.neigh + self.nbc_x - 1)/(factorial(self.neigh)*factorial(self.nbc_x - 1)))
            p_apri = np.zeros((nbconf, self.nbc_x))
        for i in range(1, X.shape[1] - 1):
            for j in range(1, X.shape[2] - 1):
                idx_vos, idx_class = self.config_voisinage(X, i, j)
                for k,e in enumerate(idx_vos):
                    p_apri[e, int(idx_class[k])] = p_apri[e, int(idx_class[k])] + 1
        p_apri = p_apri + self.reg
        p_apri = p_apri/ (p_apri.sum(axis=-1)[..., np.newaxis])
        return p_apri

    def init_kmeans(self, data):
        hidden = np.zeros(data.shape[:-1])
        test = [a.flatten() for a in np.indices(data.shape)] #Parcours ligne par ligne
        data = data[test[0], test[1]].reshape(-1,1)
        kmeans = KMeans(n_clusters=self.nbc_x).fit(data)
        hidden[test[0], test[1]] = kmeans.labels_
        self.proba = self.estim_proba_apri(hidden[np.newaxis, ...])
        hidden=hidden.flatten()
        self.mu = np.zeros((self.nbc_x,data.shape[-1]))
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
        self.sigma = self.sigma + np.full(self.sigma.shape, np.eye(self.sigma.shape[-1]) * self.reg)

    def init_Gibbs(self, shape):
        shape = (shape[0],) + tuple(x + 2 for x in shape[1:])
        nb_class = self.nbc_x
        p_tmp = np.array([1 / nb_class] * nb_class)
        x_init = np.argmax(np.random.multinomial(1, p_tmp, shape),axis=-1)
        return x_init

    def iter_Gibbs_proba(self, x_init):
        x_new = x_init
        for i in range(1, x_init.shape[1] - 1):
            for j in range(1, x_init.shape[2] - 1):
                idx_vos, _ = self.config_voisinage(x_new, i, j)
                x_new[:, i, j] = random_multinomial_vect(self.proba[idx_vos, :])
        return x_new

    def genere_Gibbs_proba(self, x_init, nb_iter):
        x = np.zeros(x_init.shape)
        x[:, :, :] = x_init
        for i in range(0, nb_iter):
            x[:,:, :] = self.iter_Gibbs_proba(x[:, :, :])
        return crop_border(x[:,:, :])

    def iter_Gibbs_proba_apost_gauss(self, gausses, x_init):
        x_new = x_init
        for i in range(1, x_init.shape[1] - 1):
            for j in range(1, x_init.shape[2] - 1):
                idx_vos, _ = self.config_voisinage(x_new, i, j)
                p_apost = self.calc_proba_champs_apost(gausses[i - 1, j - 1])
                x_new[:, i, j] = random_multinomial_vect(p_apost[idx_vos, :]/p_apost[idx_vos, :].sum(axis=1)[...,np.newaxis])
        return x_new

    def genere_Gibbs_proba_apost(self, gausses, x_init, nb_iter):
        x = np.zeros(x_init.shape)
        x[:, :, :] = x_init
        for i in range(0, nb_iter):
            x[:, :, :] = self.iter_Gibbs_proba_apost_gauss(gausses, x[:,:, :])
        return crop_border(x[:, :, :])

    def seg_mpm(self, data, iter_gibbs, nb_simu):
        gausses = self.get_gaussians(data)
        x_init = self.init_Gibbs((nb_simu,) + data.shape[:-1])
        x_simu = self.genere_Gibbs_proba_apost(gausses, x_init, iter_gibbs)
        p_apost = (x_simu[..., np.newaxis] == np.indices((self.nbc_x,))[np.newaxis,np.newaxis,...]).sum(axis=0)
        return np.argmax(p_apost, axis=-1)

    def save_param_to_json(self, filepath):
        param_s = {'proba': self.proba.tolist(),'mu': self.mu.tolist(),
                   'sig': self.sigma.tolist()}
        with open(filepath,'w') as f:
            json.dump(param_s, f, ensure_ascii=False)

    def load_param_from_json(self, filepath):
        with open(filepath, 'r') as f:
            params = json.load(f)
        self.proba = np.array(params['proba'])
        self.mu = np.array(params['mu'])
        self.sigma = np.array(params['sig'])

    def calc_param_EM(self, data, iter_gibbs, nb_simu):
        gausses = self.get_gaussians(data)
        x_init = self.init_Gibbs((nb_simu,) + data.shape[:-1])
        x_simu = self.genere_Gibbs_proba_apost(gausses, x_init, iter_gibbs)
        self.proba = self.estim_proba_apri(x_simu)
        hidden=x_simu.flatten()
        data = np.stack([data]*nb_simu, axis=0).flatten()[...,np.newaxis]
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
        self.sigma = self.sigma + np.full(self.sigma.shape, np.eye(self.sigma.shape[-1]) * self.reg)

    def get_param_EM(self, data, iter, iter_gibbs, nb_simu, early_stopping=0):
        print({'iter': 0, 'proba': self.proba, 'mu': self.mu, 'sigma': self.sigma})
        for q in range(iter):
            prev_proba = self.proba
            prev_mu = self.mu
            prev_sigma = self.sigma
            self.calc_param_EM(data, iter_gibbs, nb_simu)
            if early_stopping is not None:
                diff_log_likelihood = np.sqrt(
                    (((self.proba - prev_proba) ** 2).sum() + (
                                (self.mu - prev_mu) ** 2).sum() + (
                            (self.sigma - prev_sigma) ** 2).sum()))
                print({'iter': q + 1, 'proba': self.proba, 'mu': self.mu, 'sigma': self.sigma,
                       'diff_norm_param': diff_log_likelihood})
                if diff_log_likelihood <= early_stopping:
                    break
            else:
                print({'iter': q + 1, 'proba': self.proba, 'mu': self.mu, 'sigma': self.sigma})

