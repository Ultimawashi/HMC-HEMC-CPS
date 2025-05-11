import numpy as np
import cv2 as cv
import os
import json
from utils import get_peano_index, convert_multcls_vectors, moving_average, cut_diff, split_in, standardize_np, peano_to_neighbours
from hmm_clean import HMC_ctod, HEMC_ctod, HMC_neigh_ctod, HEMC_neigh_ctod
from hmf import HMF_ctod
from sklearn.cluster import KMeans

srcfolder = './img_reelles/'
resfolder = './test_real_gray_images'
imgfs = [
         'radio1',
         'tree1',
         ]

max_val = 255
resolutions = [(512, 512)]
kmeans_clusters = 3
models = [
    {'name': 'hmc', 'model': HMC_ctod(3, vect=True), 'params': None},
    {'name': 'hmc_neigh', 'model': HMC_neigh_ctod(3, vect=True), 'params': None},
    {'name': 'hemc', 'model': HEMC_ctod(3, u='one', vect=True), 'params': None},
    {'name': 'hemc_neigh', 'model': HEMC_neigh_ctod(3, u='one', vect=True), 'params': None},
    {'name': 'hmf', 'model': HMF_ctod(3), 'params': None}]

kmeans_clusters = 3
iterEM = 1
iterGibbs = 1
simuGibbs = 1

if not os.path.exists(resfolder):
    os.makedirs(resfolder)

for resolution in resolutions:
    for imgf in imgfs:

        img = cv.imread(srcfolder + imgf + '.bmp') # Charger l'image
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # Si cette ligne est décommentée on travaille en niveau de gris
        img = cv.resize(img, resolution)
        cv.imwrite(
            resfolder + '/' + imgf + '_' + str(resolution[0]) + '_' + str(
                resolution[1])+'.bmp',
            img)

        test = get_peano_index(img.shape[0])  # Parcours de peano

        data = img[test[0], test[1]].reshape(-1, 1)
        kmeans = KMeans(n_clusters=kmeans_clusters).fit(data)
        seg_kmeans = np.zeros(
            (img.shape[0], img.shape[1]))
        seg_kmeans[test[0], test[1]] = kmeans.labels_
        mu_kmeans = (((kmeans.labels_[..., np.newaxis] == np.indices((kmeans_clusters,)))[..., np.newaxis] * data[:, np.newaxis,
                                                                                                    ...]).sum(axis=0) /
                   (
                           kmeans.labels_[..., np.newaxis] == np.indices((kmeans_clusters,))).sum(axis=0)[
                       ..., np.newaxis])
        sigma_kmeans = (((kmeans.labels_[..., np.newaxis] == np.indices((kmeans_clusters,))).reshape(
            (kmeans.labels_.shape[0], kmeans_clusters))[..., np.newaxis, np.newaxis] * np.einsum('...i,...j',
                                                                                          (data[:, np.newaxis, ...] -
                                                                                           mu_kmeans[np.newaxis, ...]),
                                                                                          (data[:, np.newaxis, ...] -
                                                                                           mu_kmeans[
                                                                                               np.newaxis, ...]))).sum(
            axis=0)
                      / ((kmeans.labels_[..., np.newaxis] == np.indices((kmeans_clusters,))).sum(
                    axis=0)).reshape((kmeans_clusters,))[..., np.newaxis, np.newaxis])
        param_s = {'mu': mu_kmeans.tolist(),
                   'sig': sigma_kmeans.tolist()}

        with open(os.path.join(resfolder, imgf + '_' + str(resolution[0]) + '_' + str(
                resolution[1]) + '_param_kmeans.txt'),
                  'w') as f:
            json.dump(param_s, f, ensure_ascii=False)
        cv.imwrite(
            resfolder + '/' + imgf + '_' + str(resolution[0]) + '_' + str(
                resolution[1]) + '_seg_kmeans.bmp',
            seg_kmeans * int(max_val / (kmeans_clusters - 1)))
        for model in models:
            if 'mf' in model['name']:
                model['model'].init_kmeans(img[...,np.newaxis])
            else:
                model['model'].init_kmeans(data)
            if 'neigh' in model['name']:
                data_neighh, data_neighv = peano_to_neighbours(img)
                model['model'].get_param_EM(data, data_neighh, data_neighv, iterEM, early_stopping=10**-10)
                mpm_res = model['model'].seg_mpm(
                    data, data_neighh, data_neighv)
                if hasattr(model['model'], 'nbc_u'):
                    res_mpm_u = model['model'].seg_mpm_u(
                        data, data_neighh, data_neighv)
            elif 'mf' in model['name']:
                model['model'].get_param_EM(img[..., np.newaxis], iterEM, iter_gibbs=iterGibbs, nb_simu=simuGibbs,
                                            early_stopping=10 ** -10)
                mpm_res = model['model'].seg_mpm(
                    img[..., np.newaxis], iter_gibbs=iterGibbs, nb_simu=simuGibbs)[test[0], test[1]]
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
                resfolder + '/' + imgf + '_' + str(resolution[0]) + '_' + str(
                    resolution[1]) + '_seg_' + model['name'] + '.bmp',
                seg * int(max_val/(model['model'].nbc_x-1)))  # Sauvegarder l'image
            model['model'].save_param_to_json(os.path.join(resfolder, imgf + '_' + str(resolution[0]) + '_' + str(
                    resolution[1]) + '_param_' + model['name'] + '.txt'))
