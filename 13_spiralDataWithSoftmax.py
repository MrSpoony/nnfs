from ActivationReLU import ActivationReLU
from ActivationSoftmax import ActivationSoftmax
from LayerDense import LayerDense
from nnfs.datasets import spiral_data
import nnfs

nnfs.init()

X, y = spiral_data(100, 3)

dense1 = LayerDense(2, 3)
activation1 = ActivationReLU()
dense2 = LayerDense(3, 3)
activation2 = ActivationSoftmax()

dense1.forward(X)
activation1.forward(dense1.output)
dense2.forward(activation1.output)
activation2.forward(dense2.output)

if __name__ == '__main__':
    print(activation2.output[:5])
