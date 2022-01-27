import numpy as np

layer_outputs = [[4.8, 1.21, 2.385],
                 [8.9, -1.81, 0.2],
                 [1.41, 1.051, 0.026]]

exp_values = np.exp(layer_outputs)
norm_values = exp_values / np.sum(exp_values, axis=1, keepdims=True)

if __name__ == '__main__':
    print("Exponential values: ")
    print(exp_values)
    print("Normalized exponentiated values: ")
    print(norm_values)
    print("Sum of normalized values:")
    print(np.sum(norm_values))
