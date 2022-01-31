import numpy as np


class Accuracy:

    @staticmethod
    def calculate(y_predictions, y_true):
        predictions = np.argmax(y_predictions, axis=1)
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)
        return np.mean(predictions == y_true)
