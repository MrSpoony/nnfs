import numpy as np

from common.Loss import Loss


class LossCategoricalCrossEntropy(Loss):

    def __init__(self):
        self.dinputs = None

    def forward(self, y_prediction, y_true):
        samples = len(y_prediction)

        y_prediction_clipped = np.clip(y_prediction, 1e-7, 1 - 1e-7)

        # If the true values are in the format [0, 2, 3, 1]
        if len(y_true.shape) == 1:
            correct_confidences = y_prediction_clipped[range(samples), y_true]

        # If the true values are in the format [[1, 0, 0, 0],
        #                                       [0, 0, 1, 0],
        #                                       [0, 0, 0, 1],
        #                                       [0, 1, 0, 0]]
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_prediction_clipped * y_true, axis=1)
        else:
            correct_confidences = [[]]

        return -np.log(correct_confidences)

    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        labels = len(dvalues[0])

        # If the true values are in the format [0, 2, 3, 1]
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]

        self.dinputs = -y_true / dvalues

        self.dinputs = self.dinputs / samples
