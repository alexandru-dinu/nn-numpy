from fc_layer import FCLayer
from network import Network
from activation_functions import logistic_stable
import numpy as np
from loss_functions import L1Loss

"""
TODO: 
- add data loader
- test FC net
- add CONV net
"""


def train():
    net = Network([
        FCLayer(2, 1, logistic_stable)
    ])

    num_iter = int(1e5)
    lossf = L1Loss

    for i in range(num_iter):
        data = np.random.random((2, 1))
        target = int(data[0] > data[1])

        out = net.forward(data)
        loss = lossf(out, target)

        net.backward(data, np.sign(out))



if __name__ == '__main__':
    train()
