import numpy as np


def dist_quad(x, y):
    assert np.array_equal(np.array(x.shape), np.array(y.shape))
    return np.sum((x - y) ** 2)


def manhattan_dist(x, y):
    return np.sum(np.abs(x - y))


def gauss(d, sig):
    return np.exp(-((d / sig) ** 2) / 2) / sig
    # 0 1/sig

def normalized_gaussian(d, sig):
    return np.exp(-((d / sig) ** 2) / 2)
    # 0 et 1

def euclidean_norm(x, y):
    assert np.array_equal(np.array(x.shape), np.array(y.shape))
    return np.sqrt(np.sum((x - y) ** 2))


def normalized_euclidean_norm(x, y):
    assert np.array_equal(np.array(x.shape), np.array(y.shape))
    return np.sqrt(np.sum((x - y) ** 2))/np.sqrt(x.shape[0])
