import numpy as np
from math import log2
from scipy import ndimage
from scipy.stats import multivariate_normal
from sympy.utilities.iterables import multiset_permutations
from itertools import groupby
import cv2 as cv


def ln_sum(a,b):
    return np.maximum(a,b) + np.log(1 + np.exp(np.minimum(b-a,a-b)))


def ln_sum_np(iterable):
    res = np.NINF
    for i in range(len(iterable)):
        res = ln_sum(res,iterable[i])
    return res


def sigmoid_np(x):
    return 1 / (1 + np.exp(-x))


def heaviside_np(x):
    thresold = np.max(x)/2
    return (x < thresold) * 0 + (x >= thresold)


def thresolding_np(x, nb_class):
    thresold = np.max(x)/(2*(nb_class-1))
    first = [(x < thresold) * 0]
    last = [(x >= (nb_class*thresold))*(nb_class-1)]
    res = first + [np.logical_and(x >= ((i)*thresold ) , x<((i+2)*thresold ))*i for i in range(1,(nb_class-1))] + last
    return sum(res)


def standardize_np(x):
    return (x-np.mean(x))/np.std(x)


def get_peano_index(dSize):
    xTmp = 0
    yTmp = 0
    dirTmp = 0
    dirLookup = np.array(
        [[3, 0, 0, 1], [0, 1, 1, 2], [1, 2, 2, 3], [2, 3, 3, 0], [1, 0, 0, 3], [2, 1, 1, 0], [3, 2, 2, 1],
         [0, 3, 3, 2]]).T
    dirLookup = dirLookup + np.array(
        [[4, 0, 0, 4], [4, 0, 0, 4], [4, 0, 0, 4], [4, 0, 0, 4], [0, 4, 4, 0], [0, 4, 4, 0], [0, 4, 4, 0],
         [0, 4, 4, 0]]).T
    orderLookup = np.array(
        [[0, 2, 3, 1], [1, 0, 2, 3], [3, 1, 0, 2], [2, 3, 1, 0], [1, 3, 2, 0], [3, 2, 0, 1], [2, 0, 1, 3],
         [0, 1, 3, 2]]).T
    offsetLookup = np.array([[1, 1, 0, 0], [1, 0, 1, 0]])
    for i in range(int(log2(dSize))):
        xTmp = np.array([(xTmp - 1) * 2 + offsetLookup[0, orderLookup[0, dirTmp]] + 1,
            (xTmp - 1) * 2 + offsetLookup[0, orderLookup[1, dirTmp]] + 1,
            (xTmp - 1) * 2 + offsetLookup[0, orderLookup[2, dirTmp]] + 1,
            (xTmp - 1) * 2 + offsetLookup[0, orderLookup[3, dirTmp]] + 1])

        yTmp = np.array([(yTmp - 1) * 2 + offsetLookup[1, orderLookup[0, dirTmp]] + 1,
            (yTmp - 1) * 2 + offsetLookup[1, orderLookup[1, dirTmp]] + 1,
            (yTmp - 1) * 2 + offsetLookup[1, orderLookup[2, dirTmp]] + 1,
            (yTmp - 1) * 2 + offsetLookup[1, orderLookup[3, dirTmp]] + 1])

        dirTmp = np.array([dirLookup[0, dirTmp],dirLookup[1, dirTmp], dirLookup[2, dirTmp], dirLookup[3, dirTmp]])

        xTmp = xTmp.T.flatten()
        yTmp = yTmp.T.flatten()
        dirTmp = dirTmp.flatten()

    x = - xTmp
    y = - yTmp
    return x,y


def get_peano_neighbours_index(x, y):
    # fonction qui récupère les indices des voisins hors chaînes de markov
    indexes = []
    tmp = []
    comptmp = []
    comptmp.append((x[1], y[1]))
    n = int(np.sqrt(len(x))) - 1

    if (1, 0) in comptmp:
        tmp.append([0, 1])
    else:
        tmp.append([1, 0])
    indexes.append(tmp)

    for i in range(1, len(x) - 1):
        tmp = []
        comptmp = []
        comptmp.append([x[i - 1], y[i - 1]])
        comptmp.append([x[i + 1], y[i + 1]])
        if [x[i] + 1, y[i]] not in comptmp and x[i] != n:
            tmp.append([x[i] + 1, y[i]])
        if [x[i] - 1, y[i]] not in comptmp and x[i] != 0:
            tmp.append([x[i] - 1, y[i]])
        if [x[i], y[i] + 1] not in comptmp and y[i] != n:
            tmp.append([x[i], y[i] + 1])
        if [x[i], y[i] - 1] not in comptmp and y[i] != 0:
            tmp.append([x[i], y[i] - 1])
        indexes.append(tmp)

    tmp = []
    comptmp = []
    comptmp.append([x[n - 1], y[n - 1]])
    if [0, n - 1] in comptmp:
        tmp.append([0, n - 1])
    else:
        tmp.append([1, n])
    indexes.append(tmp)
    return indexes


def peano_to_neighbours(img):
    # fonction qui prend en argument une image et qui renvoie deux listes constituées des voisins hors chaîne de markov
    # Lorsqu'un pixel ne possède pas 2 voisins hors chaîne de Markov, on remplace les voisins par la valeur NaN
    (x, y) = get_peano_index(img.shape[0])

    indexes = get_peano_neighbours_index(x, y)
    neighbours = []
    for i in range(len(indexes)):
        tmp = np.full(2, float("NaN"), dtype='float64')
        t = 0
        for j in indexes[i]:
            tmp[t] = img[j[0], j[1]]
            t += 1
        neighbours.append(tmp)

    neighbours = np.array(neighbours)
    neighboursh, neighboursv = np.hsplit(neighbours, 2)

    return neighboursh, neighboursv


def convert_multcls_vectors(data, rand_vect_param):
    classes = range(np.max(data).astype(int) + 1)
    assert (len(classes) <= np.prod(rand_vect_param)), 'Les paramètres du vecteur aléatoire ne correspondent pas'
    res = np.zeros(data.shape + (len(rand_vect_param,)))
    aux = [convertcls_vect(cls, rand_vect_param) for cls in classes]
    for c in classes:
        res[data==c] = aux[c]
    return res.astype('int')


def convertcls_vect(cls, rand_vect_param):
    aux = cls
    res=np.zeros((len(rand_vect_param)))
    for i in reversed(range(len(rand_vect_param))):
        res[len(rand_vect_param) - i - 1] = aux % rand_vect_param[i]
        aux = aux // rand_vect_param[len(rand_vect_param) - i - 1]
    return res


def convert_vect_multcls(data, rand_vect_param):
    vectors = np.stack([a.flatten() for a in reversed(np.indices(rand_vect_param))]).T
    res = np.zeros(data.shape[:-1])
    for i,v in enumerate(vectors):
        res[(data == v).all(axis=-1)] = i
    return res.astype('int')


def convert_vect_multcls_no_order(data, rand_vect_param):
    vectors = np.stack([a.flatten() for a in reversed(np.indices(rand_vect_param))]).T
    vectors = np.unique(np.sort(vectors,axis=-1), axis=0)
    data = np.sort(data,axis=-1)
    res = np.zeros(data.shape[:-1])
    for i,v in enumerate(vectors):
        res[(data == v).all(axis=-1)] = i
    return res.astype('int')


def np_multivariate_normal_pdf(x, mu, cov):
    # if mu.shape[0] != x.shape[0]:
    #     broadc = (len(mu.shape) - len(x.shape) + 1)
    #     x = x.reshape((x.shape[0],) + (1,) * broadc + x.shape[1:])
    # else:
    #     broadc = (len(mu.shape) - len(x.shape))
    #     x = x.reshape((x.shape[0],) + (1,) * broadc + x.shape[1:])
    broadc = (len(mu.shape) - 1)
    x = x.reshape(x.shape[:-1] + (1,) * broadc + (x.shape[-1],))
    part1 = 1 / (((2 * np.pi) ** (mu.shape[-1] / 2)) * (np.linalg.det(cov) ** (1 / 2)))
    part2 = (-1 / 2) * np.einsum('...j,...j',np.einsum('...j,...ji',(x - mu),np.linalg.inv(cov)),(x - mu))
    return part1 * np.exp(part2)


def np_multivariate_normal_pdf_marginal(x, mu, cov, j, i=0):
    # if mu.shape[0] != x.shape[0]:
    #     broadc = (len(mu.shape) - len(x.shape) + 1)
    #     x = x.reshape((x.shape[0],) + (1,) * broadc + x.shape[1:])
    # else:
    #     broadc = (len(mu.shape) - len(x.shape))
    #     x = x.reshape((x.shape[0],) + (1,) * broadc + x.shape[1:])
    broadc = (len(mu.shape) - len(x.shape) + 1)
    x = x.reshape((x.shape[0],) + (1,) * broadc + x.shape[1:])
    part1 = 1 / (((2 * np.pi) ** (mu[...,i:j+1].shape[-1] / 2)) * (np.linalg.det(cov[...,i:j+1,i:j+1]) ** (1 / 2)))
    part2 = (-1 / 2) * np.einsum('...j,...j',np.einsum('...j,...ji',(x - mu[...,i:j+1]),np.linalg.inv(cov[...,i:j+1,i:j+1])),(x - mu[...,i:j+1]))
    return part1 * np.exp(part2)


def multinomial_rvs(n, p):
    """
    Sample from the multinomial distribution with multiple p vectors.

    * count must be an (n-1)-dimensional numpy array.
    * p must an n-dimensional numpy array, n >= 1.  The last axis of p
      holds the sequence of probabilities for a multinomial distribution.

    The return value has the same shape as p.
    """

    count = np.full(p.shape[:-1], n)
    out = np.zeros(p.shape, dtype=int)
    ps = p.cumsum(axis=-1)
    # Conditional probabilities
    with np.errstate(divide='ignore', invalid='ignore'):
        condp = p / ps
    condp[np.isnan(condp)] = 0.0
    for i in range(p.shape[-1]-1, 0, -1):
        binsample = np.random.binomial(count, condp[..., i])
        out[..., i] = binsample
        count -= binsample
    out[..., 0] = count
    return out


def moving_average(x, param, neighbour=4):
    assert (neighbour == 4 or neighbour == 8), 'please choose only between 4 and 8 neighbour'
    x = x.astype(float)
    if neighbour == 4:
        kernel = np.array([[0, param, 0], [param, 1, param], [0, param, 0]])
    else:
        kernel = np.array([[param, param, param], [param, 1,param], [param, param, param]])
    res = ndimage.convolve(x, kernel, mode='reflect')
    return res


def calc_product(list_mat):
    res=list_mat[len(list_mat)-1]
    for i in reversed(range(len(list_mat)-1)):
        res=(res.flatten()*list_mat[i].T).T
    return res.reshape(int(np.sqrt(np.prod(res.shape))), int(np.sqrt(np.prod(res.shape))))


def algorithm_u(ns, m):
    def visit(n, a):
        ps = [[] for i in range(m)]
        for j in range(n):
            ps[a[j + 1]].append(ns[j])
        return ps

    def f(mu, nu, sigma, n, a):
        if mu == 2:
            yield visit(n, a)
        else:
            for v in f(mu - 1, nu - 1, (mu + sigma) % 2, n, a):
                yield v
        if nu == mu + 1:
            a[mu] = mu - 1
            yield visit(n, a)
            while a[nu] > 0:
                a[nu] = a[nu] - 1
                yield visit(n, a)
        elif nu > mu + 1:
            if (mu + sigma) % 2 == 1:
                a[nu - 1] = mu - 1
            else:
                a[mu] = mu - 1
            if (a[nu] + sigma) % 2 == 1:
                for v in b(mu, nu - 1, 0, n, a):
                    yield v
            else:
                for v in f(mu, nu - 1, 0, n, a):
                    yield v
            while a[nu] > 0:
                a[nu] = a[nu] - 1
                if (a[nu] + sigma) % 2 == 1:
                    for v in b(mu, nu - 1, 0, n, a):
                        yield v
                else:
                    for v in f(mu, nu - 1, 0, n, a):
                        yield v

    def b(mu, nu, sigma, n, a):
        if nu == mu + 1:
            while a[nu] < mu - 1:
                yield visit(n, a)
                a[nu] = a[nu] + 1
            yield visit(n, a)
            a[mu] = 0
        elif nu > mu + 1:
            if (a[nu] + sigma) % 2 == 1:
                for v in f(mu, nu - 1, 0, n, a):
                    yield v
            else:
                for v in b(mu, nu - 1, 0, n, a):
                    yield v
            while a[nu] < mu - 1:
                a[nu] = a[nu] + 1
                if (a[nu] + sigma) % 2 == 1:
                    for v in f(mu, nu - 1, 0, n, a):
                        yield v
                else:
                    for v in b(mu, nu - 1, 0, n, a):
                        yield v
            if (mu + sigma) % 2 == 1:
                a[nu - 1] = 0
            else:
                a[mu] = 0
        if mu == 2:
            yield visit(n, a)
        else:
            for v in b(mu - 1, nu - 1, (mu + sigma) % 2, n, a):
                yield v

    n = len(ns)
    a = [0] * (n + 1)
    for j in range(1, m + 1):
        a[n - m + j] = j - 1
    return f(m, n, 0, n, a)


def calc_err(ref_im, seg_im):
    terr = np.sum(seg_im != ref_im) / np.prod(ref_im.shape)
    return (terr <= 0.5) * terr + (terr > 0.5) * (1 - terr)


def calc_err_v2(ref_im, seg_im):
    class_colors=np.unique(seg_im)
    for e in range(len(class_colors)):
        seg_im[seg_im==class_colors[e]]=e
    terr = 1
    for p in multiset_permutations(class_colors):
        seg_im_prime = np.choose(seg_im, np.array(p))
        terrprime = np.sum(seg_im_prime != ref_im) / np.prod(ref_im.shape)
        if terrprime < terr:
            terr = terrprime
    return terr


def split_in(num, list):
    return [list[x:x + num] for x in range(0, len(list), num)]


def cut_diff(inp):
    return [list(g) for k, g in groupby(inp, key=lambda i: i)]


def calc_matDS(m, lx):
    nbc_x, nbc_u = lx.shape
    return np.moveaxis((m[np.newaxis,:,:]*lx[:,:,np.newaxis]).reshape(nbc_u*nbc_x, nbc_u)[np.newaxis,...]*lx[:,np.newaxis,:],0,1).reshape(nbc_u*nbc_x, nbc_u*nbc_x)


def calc_transDS(m, lx):
    nbc_x, nbc_u = lx.shape
    return np.tile(np.moveaxis((m[np.newaxis,:,:]*lx[:,np.newaxis,:]),0,1).reshape(nbc_u, nbc_u*nbc_x), (nbc_x,1))


def calc_vectDS(pm,lx):
    nbc_x, nbc_u = lx.shape
    return (pm[np.newaxis,...]*lx).reshape((nbc_u*nbc_x))


def pad_gray_im_to_square(im):
    height, width = im.shape
    x = height if height > width else width
    y = height if height > width else width
    square = np.zeros((x, y), np.uint8)
    square[int((y - height) / 2):int(y - (y - height) / 2), int((x - width) / 2):int(x - (x - width) / 2)] = im
    return square


def resize_gray_im_to_square(im):
    height, width = im.shape
    x = height if height > width else width
    y = height if height > width else width
    return cv.resize(im, (x,y))


def calc_cacheDS(lx, hidden, nbc_u1):
    nbc_u2 = lx.shape[1]
    res = np.zeros((hidden.shape[0],nbc_u2))
    for i in range(hidden.shape[0]):
        res[i] = np.any(lx[hidden[i]:(hidden[i]+1)*nbc_u1],axis=0)
    return res


def non_stationary_noise(image, mu, sigma, a = 1):
    res = np.zeros(image.shape + (mu.shape[-1],))
    for i in range(0, res.shape[0]):
        for j in range(0, res.shape[1]):
            new_mu_ij = np.random.uniform(low=mu[image[i,j]] - a*(np.linalg.cholesky(sigma[image[i,j]])/2), high=mu[image[i,j]] + a*(np.linalg.cholesky(sigma[image[i,j]])/2), size=None)
            res[i,j] = multivariate_normal.rvs(new_mu_ij, sigma[image[i,j]])
    return res


def add_border(img, size=1):
    """

    :param img:
    :param size:
    :return:
    """
    new_img = np.zeros(tuple(x + 2*size for x in img.shape))
    new_img[size:-size,size:-size] = img
    return new_img


def crop_border(img, size=1):
    return img[...,size:-size,size:-size]


def random_multinomial_vect(prob_matrix):
    items = np.arange(0, prob_matrix.shape[1])
    s = prob_matrix.cumsum(axis=1)
    r = np.random.rand(prob_matrix.shape[0])
    k = (s < r[...,np.newaxis]).sum(axis=1)
    k[k>=prob_matrix.shape[1]] = (prob_matrix.shape[1]-1)
    return items[k]