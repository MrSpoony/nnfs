import numpy as np

dvalues = np.array([[1., 1., 1.],
                    [2., 2., 2.],
                    [3., 3., 3.]])

weights = np.array([[0.2, 0.8, -0.5, 1],
                    [0.5, -0.91, -0.26, -0.5],
                    [-0.26, -0.27, 0.17, 0.87]]).T

inputs = np.array([[1., 2., 3., 2.5],
                   [2., 5., -1., 2.],
                   [1.5, 2.7, 3.3, -0.8]])

biases = np.array([[2., 3., 0.5]])

z = np.array([[1., 2., -3],
              [2., -7., -1.],
              [-1., 2., 5,]])

dvalues = np.array([[1., 2., 3.],
                    [4., 5., 6.],
                    [7., 8., 9.]])

drelu = dvalues.copy()
drelu[z <= 0] = 0

dinputs = np.dot(dvalues, weights.T)
dweights = np.dot(inputs.T, dvalues)
dbiases = np.sum(dvalues, axis=0, keepdims=True)

if __name__ == '__main__':
    print(dinputs)
    print()
    print(dweights)
    print()
    print(dbiases)
    print()
    print(drelu)
