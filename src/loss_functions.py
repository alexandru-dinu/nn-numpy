import numpy as np


def L1Loss(x, y):
    return np.mean(np.abs(x - y))


def MSELoss(x, y):
    return np.mean((x - y) ** 2)


def BCELoss(x, y):
    return -(y * np.log(x) + (1 - y) * np.log(1-x))