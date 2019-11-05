import numpy as np


def identity(x, dx=False):
    return x if not dx else np.ones(x.shape)


def sigmoid(x, dx=False):
    y = 1 / (1 + np.exp(-x))
    return y if not dx else y * (1 - y)


def sigmoid_s(x, dx=False):
    return sigmoid(np.clip(x, -10, 10), dx)


def tanh(x, dx=False):
    y = (np.exp(2 * x) - 1) / (np.exp(2 * x) + 1)
    return y if not dx else 1 - y * y


def tanh_s(x, dx=False):
    return tanh(np.clip(x, -10, 10), dx)


def relu(x, dx=False):
    if dx:
        x[x > 0] = 1
    x[x <= 0] = 0
    return x


# TODO
def softmax(x):
    pass