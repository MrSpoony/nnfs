import numpy as np


class LayerDense:

    def __init__(self, n_inputs, n_neurons):
        """

        Args:
            n_inputs: The number of inputs this layer has
            n_neurons: The number of neurons this layer has
        """
        self.output = None
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
        self.n_inputs = n_inputs
        self.n_neurons = n_neurons

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights)
