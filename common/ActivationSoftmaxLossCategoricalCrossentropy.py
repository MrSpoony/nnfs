import numpy as np

from common.ActivationSoftmax import ActivationSoftmax
from common.LossCategoricalCrossEntropy import LossCategoricalCrossEntropy


class ActivationSoftmaxLossCategoricalCrossentropy():

    def __init__(self):
        self.output = None
        self.activation = ActivationSoftmax()
        self.loss = LossCategoricalCrossEntropy()

    def forward(self, inputs, y_true):
        self.activation.forward(inputs)
        self.output = self.activation.output

        return self.loss.calculate(self.output, y_true)

    def backward(self, dvalues, y_true):
        samples = len(dvalues)

        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)

        self.dinputs = dvalues.copy()

        self.dinputs[range(samples), y_true] -= 1

        self.dinputs = self.dinputs / samples

