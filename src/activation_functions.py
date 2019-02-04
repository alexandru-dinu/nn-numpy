import numpy as np


def identity(x, derivate=False):
    return x if not derivate else np.ones(x.shape)


def logistic(x, derivative=False):
    y = 1 / (1 + np.exp(-x))

    return y if not derivative else y * (1 - y)


def logistic_stable(x, derivative=False):
    x[x < -10] = -10
    x[x > 10] = 10

    return logistic(x, derivative)


def hyperbolic_tangent(x, derivative=False):
    y = (np.exp(2 * x) - 1) / (np.exp(2 * x) + 1)

    return y if not derivative else 1 - y * y


def hyperbolic_tangent_stable(x, derivative=False):
    x[x < -10] = -10
    x[x > 10] = 10

    return hyperbolic_tangent(x, derivative)


def relu(x, derivative=False):
    if derivative:
        x[x > 0] = 1

    x[x <= 0] = 0

    return x
