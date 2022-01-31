from nnfs.datasets import spiral_data
import nnfs

from common.Accuracy import Accuracy
from common.ActivationReLU import ActivationReLU
from common.ActivationSoftmax import ActivationSoftmax
from common.LayerDense import LayerDense
from common.LossCategoricalCrossEntropy import LossCategoricalCrossEntropy

nnfs.init()

X, y = spiral_data(100, 3)

dense1 = LayerDense(2, 3)
activation1 = ActivationReLU()
dense2 = LayerDense(3, 3)
activation2 = ActivationSoftmax()

lossFunction = LossCategoricalCrossEntropy()
accuracyFunction = Accuracy()


dense1.forward(X)
activation1.forward(dense1.output)
dense2.forward(activation1.output)
activation2.forward(dense2.output)

loss = lossFunction.calculate(activation2.output, y)
accuracy = accuracyFunction.calculate(activation2.output, y)


if __name__ == '__main__':
    print(activation2.output[:5])
    print('loss', loss)
    print('accuracy', accuracy)
