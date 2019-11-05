import numpy as np
from layer_interface import LayerInterface


class Dense(LayerInterface):
    def __init__(self, num_inputs, num_outputs, activation):
        """
        z = Wx + b
        y = f(z)
        """
        
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.f = activation

        self.weights = np.random.normal(
            0,
            np.sqrt(2.0 / (self.num_inputs + self.num_outputs)),
            (self.num_outputs, self.num_inputs)
        )

        self.biases = np.random.normal(
            0,
            np.sqrt(2.0 / (self.num_inputs + self.num_outputs)),
            (self.num_outputs, 1)
        )

        self.z = np.zeros((self.num_outputs, 1))
        self.outputs = np.zeros((self.num_outputs, 1))

        # gradients
        self.grad_weights = np.zeros(self.weights.shape)
        self.grad_biases = np.zeros(self.biases.shape)

    def forward(self, x):
        assert (x.shape == (self.num_inputs, 1)) # TODO

        self.z = np.dot(self.weights, x) + self.biases
        self.outputs = self.f(self.z)

        return self.outputs

    def backward(self, x, upstream):
        # return upstream * local
        aux = upstream * self.f(self.z, True)

        self.grad_biases = aux

        self.grad_weights = np.dot(x, aux.T).T

        grad_inputs = np.dot(self.weights.T, aux).T

        return grad_inputs

    def update_parameters(self, alpha):
        self.weights -= alpha * self.grad_weights
        self.biases -= alpha * self.grad_biases

    def __str__(self):
        return "[Dense (%s -> %s) | %s]" % (self.num_inputs, self.num_outputs, self.f.__name__)
    
    def __repr__(self):
        return str(self)
