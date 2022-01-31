import numpy as np

from common.LossCategoricalCrossEntropy import LossCategoricalCrossEntropy

softmax_outputs = np.array([[0.7, 0.1, 0.2],
                            [0.1, 0.5, 0.4],
                            [0.02, 0.9, 0.08]])

class_targets = np.array([[1, 0, 0],
                          [0, 1, 0],
                          [0, 1, 0]])

loss_function = LossCategoricalCrossEntropy()
loss = loss_function.calculate(softmax_outputs, class_targets)

if __name__ == '__main__':
    print(loss)
