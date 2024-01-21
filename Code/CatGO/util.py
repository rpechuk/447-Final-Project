import numpy as np

def log_likelihood(p):
    return -np.sum(np.log(p))

def normalize(array, axis=1):
    denoms = np.sum(array, axis=axis)
    if axis == 1:
        return array / denoms[:,np.newaxis]
    if axis == 0:
        return array / denoms[np.newaxis, :]

def proto_only(models):
    for model in models:
        if 'prototype' not in model:
            return False
    return True