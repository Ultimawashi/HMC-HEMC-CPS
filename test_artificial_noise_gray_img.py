import numpy as np
import cv2 as cv
import os
import json
from utils import get_peano_index, convert_multcls_vectors, moving_average, cut_diff, split_in, sigmoid_np, heaviside_np, peano_to_neighbours, non_stationary_noise
from sklearn.cluster import KMeans
from hmm_clean import HMC_ctod,  HEMC_ctod,  HMC_neigh_ctod, HEMC_neigh_ctod
from hmf import HMF_ctod


srcfolder = './img/'
resfolder = './res_test_scps_em'
imgfs = ['spaghetti', 'promenade2', 'squares', 'zebre2', 'tree','nazca', 'zebre3-beee', 'digital', 'cible_alt_zebre2_half']
resolutions = [(256,256)]
max_val = 255
gauss_noise = [

               {'ns':False,'corr': False, 'mu1': 0, 'mu2': 1, 'sig1': 1, 'sig2': 1, 'corr_param': None},
               {'ns':False,'corr': False, 'mu1': 0, 'mu2': 2, 'sig1': 1, 'sig2': 1, 'corr_param': None},
               ]
models = [
    {'name': 'hmc', 'model': HMC_ctod(2, vect=True), 'params': None},
    {'name': 'hmc_neigh', 'model': HMC_neigh_ctod(2, vect=True), 'params': None},
    {'name': 'hemc', 'model': HEMC_ctod(2, u='one', vect=True), 'params': None},
    {'name': 'hemc_neigh', 'model': HEMC_neigh_ctod(2, u='one', vect=True), 'params': None},
    {'name': 'hmf', 'model': HMF_ctod(2), 'params': None}
          ]
kmeans_clusters = 2
iterEM = 100
iterGibbs = 100
simuGibbs = 10

for resolution in resolutions:
    for imgf in imgfs:
        if not os.path.exists(resfolder + '/' + imgf):
            os.makedirs(resfolder + '/' + imgf)

        img = cv.imread(srcfolder + imgf + '.bmp')  # Charger l'image
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # Si cette ligne est décommentée on travaille en niveau de gris
        img = cv.resize(img, resolution)
        img = heaviside_np(img)
        cv.imwrite(resfolder + '/' + imgf + '/' + str(resolution[0]) + '_' + str(resolution[1]) + '.bmp', img * max_val)
        test = get_peano_index(img.shape[0])  # Parcours de peano
        # test = [a.flatten() for a in np.indices(resolution)] #Parcours ligne par ligne
        hidden = img[test[0], test[1]]

        for noise in gauss_noise:
            if not noise['ns']:
                img_noisy = (img == 0) * np.random.normal(noise['mu1'], np.sqrt(noise['sig1']), img.shape) + (
                        img == 1) * np.random.normal(noise['mu2'], np.sqrt(noise['sig2']),
                                                     img.shape)
                ns = ''
                ns_param = ''
            else:
                img_noisy = non_stationary_noise(img, np.array([[noise['mu1']], [noise['mu2']]]), np.array([[[noise['sig1']]], [[noise['sig2']]]]), a=noise['ns_param'])
                ns = 'ns'
                ns_param = str(noise['ns_param'])
            corr = ''
            corr_param = ''
            if noise['corr']:
                img_noisy = moving_average(img_noisy, noise['corr_param'])
                corr = 'corr'
                corr_param = str(noise['corr_param'])

            noise_param = '(' + str(noise['mu1']) + ',' + str(noise['sig1']) + ')' + '_' + '(' + str(
                noise['mu2']) + ',' + str(noise['sig2']) + ')'
            cv.imwrite(
                resfolder + '/' + imgf + '/' + str(resolution[0]) + '_' + str(
                    resolution[1]) + '_' + ns +  '_' + ns_param + '_' + corr + '_' + corr_param + noise_param + '.bmp', sigmoid_np(img_noisy)*max_val)

            data = img_noisy[test[0], test[1]].reshape(-1, 1)
            kmeans = KMeans(n_clusters=kmeans_clusters).fit(data)
            seg_kmeans = np.zeros(
                (img.shape[0], img.shape[1]))
            seg_kmeans[test[0], test[1]] = kmeans.labels_
            cv.imwrite(resfolder + '/' + imgf + '/' + str(resolution[0]) + '_' + str(
                resolution[1]) + '_' + ns + '_' + ns_param + '_' + corr + '_' + corr_param + noise_param + '_seg_kmeans' + '.bmp', seg_kmeans * int(max_val/(kmeans_clusters-1)))

            for model in models:
                if not model['params']:

                    if 'mf' in model['name']:

                        model['model'].init_kmeans(img_noisy[...,np.newaxis])
                    else:
                        model['model'].init_kmeans(data)
                else:
                    model['model'].load_param_from_json(model['params'])

                if 'neigh' in model['name']:
                    data_neighh, data_neighv = peano_to_neighbours(img_noisy)
                    model['model'].get_param_EM(data, data_neighh, data_neighv, iterEM, early_stopping=10**-10)
                    mpm_res = model['model'].seg_mpm(
                        data, data_neighh, data_neighv)
                    if hasattr(model['model'], 'nbc_u'):
                        res_mpm_u = model['model'].seg_mpm_u(
                        data, data_neighh, data_neighv)
                elif 'mf' in model['name']:
                    model['model'].get_param_EM(img_noisy[...,np.newaxis], iterEM, iter_gibbs=iterGibbs, nb_simu=simuGibbs,  early_stopping=10 ** -10)
                    mpm_res = model['model'].seg_mpm(
                        img_noisy[...,np.newaxis], iter_gibbs=iterGibbs, nb_simu=simuGibbs)[test[0], test[1]]
                else:
                    model['model'].get_param_EM(data, iterEM, early_stopping=10**-10)
                    mpm_res = model['model'].seg_mpm(
                        data)
                    if hasattr(model['model'], 'nbc_u'):
                        res_mpm_u = model['model'].seg_mpm_u(
                            data)
                seg = np.zeros(
                    (img.shape[0], img.shape[1]))  # Création d'une matrice vide qui va recevoir l'image segmentée
                seg[test[0], test[1]] = mpm_res  # Remplir notre matrice avec les valeurs de la segmentation
                cv.imwrite(
                    resfolder + '/' + imgf + '/' + str(resolution[0]) + '_' + str(
                        resolution[1]) + '_' + ns +  '_' + ns_param + '_' + corr + '_' + corr_param + noise_param + '_seg_' + model['name'] + '.bmp',
                    seg * max_val)  # Sauvegarder l'image
                if hasattr(model['model'], 'nbc_u'):
                    seg_u = np.zeros(
                        (img.shape[0], img.shape[1]))  # Création d'une matrice vide qui va recevoir l'image segmentée
                    seg_u[test[0], test[1]] = res_mpm_u
                    cv.imwrite(
                        resfolder + '/' + imgf + '/' + str(resolution[0]) + '_' + str(
                            resolution[1]) + '_' + ns +  '_' + ns_param + '_' + corr + '_' + corr_param + noise_param + '_smg_u_' + model[
                            'name'] + '.bmp',
                        seg_u * int(max_val/(model['model'].nbc_u-1)))  # Sauvegarder l'image
                model['model'].save_param_to_json(resfolder + '/' + imgf + '/' + str(resolution[0]) + '_' + str(
                            resolution[1]) + '_' + ns +  '_' + ns_param + '_' + corr + '_' + corr_param + noise_param + '_param_' + model[
                            'name'] + '.txt')
