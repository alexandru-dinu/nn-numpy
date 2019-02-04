import numpy as np


def L1Loss(x, y):
    return np.abs(x - y)


def MSELoss(x, y):
    return np.power(x - y, 2)
