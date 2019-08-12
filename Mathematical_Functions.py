import numpy as np


def sigmoid(x):
    return 1/(1 + np.exp(-x))


def dsigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))


def quadratic_loss(actual, expected):
    return (np.sum((actual - expected))**2)/np.size(actual)


v_sigmoid = np.vectorize(sigmoid)
v_dsigmoid = np.vectorize(dsigmoid)
