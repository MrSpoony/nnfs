import numpy as np


class LayerDense:

    def __init__(self, n_inputs, n_neurons):
        """

        Args:
            n_inputs: The number of inputs this layer has
            n_neurons: The number of neurons this layer has
        """
        self.dweights = None
        self.dinputs = None
        self.dbiases = None

        self.inputs = None
        self.output = None

        self.n_inputs = n_inputs
        self.n_neurons = n_neurons

        self.weights = 0.01 * np.random.randn(self.n_inputs, self.n_neurons)
        self.biases = np.zeros((1, self.n_neurons))

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(self.inputs, self.weights)

    def backward(self, dvalues):
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dinputs = np.dot(dvalues, self.weights.T)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)

