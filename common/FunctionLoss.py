import numpy as np


class FunctionLoss:
    def __init__(self):
        self.mean = None
        self.output = None

    def calculate(self, inputs, expected_output):
        inputs = np.array(inputs)
        expected_output = np.array(expected_output)

        correct_confidences = []

        # If expected_output is in form [0, 4, 2, 3, 1]
        if len(expected_output.shape) == 1:
            correct_confidences = inputs[[range(len(inputs))], expected_output]

        # If expected_output is in form [[0, 0, 0, 0, 1],
        #                                [0, 0, 1, 0, 0],
        #                                [0, 0, 0, 1, 0],
        #                                [0, 1, 0, 0, 0]]
        elif len(expected_output.shape) == 2:
            correct_confidences = np.sum(inputs * expected_output, axis=1)

        self.output = -np.log(correct_confidences)
        self.mean = np.mean(self.output)
