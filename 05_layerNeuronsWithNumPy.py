import numpy as np

inputs = [1.0, 2.0, 3.0, 2.5]

weights = [[0.2, 0.8, -0.5, 1.0],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, -0.17, 0.87]]
biases = [2.0, 3.0, 0.5]

layer_outputs = []
for neuron_weights, neuron_bias in zip(weights, biases):
    layer_outputs.append(np.round(np.dot(inputs, neuron_weights) + neuron_bias, 3))
layer_outputs = np.array(layer_outputs)

outputs = np.dot(weights, inputs) + biases


if __name__ == "__main__":
    print(layer_outputs)
    print(outputs)
