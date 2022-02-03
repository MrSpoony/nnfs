import numpy as np

# Passed in gradient from next layer
dvalues = np.array([[1., 1., 1.],
                    [2., 2., 2.],
                    [3., 3., 3.]])

# Batch of inputs
inputs = np.array([[1., 2., 3., 3.5],
                   [2., 5., -1., 2.],
                   [-1.5, 2.7, 3.3, -0.8]])

# 3 Sets of weights one for each neuron
weights = np.array([[0.2, 0.8, -0.5, 1],
                    [0.5, -0.91, 0.26, -0.5],
                    [-0.26, -0.27, -0.17, 0.87]]).T  # Transposed

biases = np.array([[2, 3, 0.5]])

# Forward pass
layerOutputs = np.dot(inputs, weights) + biases
reluOutputs = np.maximum(0, layerOutputs)

drelu = reluOutputs.copy()
drelu[layerOutputs <= 0] = 0

dinputs = np.dot(drelu, weights.T)
dweights = np.dot(inputs.T, dvalues)

dbiases = np.sum(drelu, axis=0, keepdims=True)

weights += -0.001 * dweights
biases += -0.001 * dbiases


if __name__ == '__main__':
    print(weights)
    print(biases)
