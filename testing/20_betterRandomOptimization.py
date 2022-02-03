import numpy as np
from nnfs.datasets import vertical_data
import nnfs

from common.Accuracy import Accuracy
from common.ActivationReLU import ActivationReLU
from common.ActivationSoftmax import ActivationSoftmax
from common.LayerDense import LayerDense
from common.LossCategoricalCrossEntropy import LossCategoricalCrossEntropy

nnfs.init()

X, y = vertical_data(100, 3)


dense1 = LayerDense(2, 3)
activation1 = ActivationReLU()
dense2 = LayerDense(3, 3)
activation2 = ActivationSoftmax()
lossFunction = LossCategoricalCrossEntropy()
accuracyFunction = Accuracy()
lowestLoss = np.inf

bestDense1Weights = dense1.weights.copy()
bestDense1Biases = dense1.biases.copy()
bestDense2Weights = dense2.weights.copy()
bestDense2Biases = dense2.biases.copy()

if __name__ == '__main__':

    for i in range(1000000):
        dense1.weights += 0.5 * np.random.randn(2, 3)
        dense1.biases += 0.5 * np.random.randn(1, 3)
        dense2.weights += 0.5 * np.random.randn(3, 3)
        dense2.biases += 0.5 * np.random.randn(1, 3)

        dense1.forward(X)
        activation1.forward(dense1.output)
        dense2.forward(activation1.output)
        activation2.forward(dense2.output)

        loss = lossFunction.calculate(activation2.output, y)

        accuracy = accuracyFunction.calculate(activation2.output, y)

        if loss < lowestLoss:
            print("New set of weights found, iteration", i, "loss:", loss, "acc:", accuracy)
            bestDense1Weights = dense1.weights.copy()
            bestDense1Biases = dense1.biases.copy()
            bestDense2Weights = dense2.weights.copy()
            bestDense2Biases = dense2.biases.copy()
            lowestLoss = loss
        else:
            dense1.weights = bestDense1Weights
            dense1.biases = bestDense1Biases
            dense2.weights = bestDense2Weights
            dense2.biases = bestDense2Biases
