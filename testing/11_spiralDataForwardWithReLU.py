from ActivationReLU import ActivationReLU
from LayerDense import LayerDense
from nnfs.datasets import spiral_data
import nnfs

nnfs.init()

X, y = spiral_data(100, 3)

dense1 = LayerDense(2, 3)
activation1 = ActivationReLU()

dense1.forward(X)
activation1.forward(dense1.output)

if __name__ == '__main__':
    print(activation1.output[:5])
