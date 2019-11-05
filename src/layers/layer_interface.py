import numpy as np


class LayerInterface:

    def __init__(self, num_inputs, num_outputs, activation):
        self.outputs = np.array([])

    def forward(self, x):
        raise NotImplementedError

    def backward(self, x, upstream):
        raise NotImplementedError

    def update_parameters(self, alpha):
        pass

    def __str__(self):
        pass
    
    def __repr__(self):
        return str(self)