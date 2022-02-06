import numpy as np


class ActivationSoftmax:
    def __init__(self):
        self.output = np.empty_like((0, 0))

    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        self.output = exp_values / np.sum(exp_values, axis=1, keepdims=True)

    def bacward(self, dvalues):
        self.dinputs = np.empty_like(dvalues)

        for i, (singleOutput, singleDvalues) in enumerate(zip(self.output, dvalues)):
            singleOutput = singleOutput.reshape(-1, 1)

            jacobianMatrix = (np.diagflat(singleOutput) -
                              np.dot(singleOutput, singleOutput.T))

            self.dinputs[i] = np.dot(jacobianMatrix, singleDvalues)
