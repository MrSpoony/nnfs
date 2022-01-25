from LayerDense import LayerDense
from nnfs.datasets import spiral_data
import nnfs

nnfs.init()

X, y = spiral_data(samples=100, classes=3)

dense1 = LayerDense(2, 3)
dense1.forward(X)

if __name__ == '__main__':
    print(dense1.output[:5])
