import numpy as np


softmax_outputs = [0.7, 0.1, 0.2]

softmax_outputs = np.array(softmax_outputs).reshape(-1, 1)
# has the same effect as
# softmax_outputs = np.array([softmax_outputs]).T

softmax_outputs_diagflat = np.diagflat(softmax_outputs)

softmax_outputs_dot = np.dot(softmax_outputs, softmax_outputs.T)
softmax_outupts_sub = softmax_outputs_diagflat - softmax_outputs_dot

if __name__ == '__main__':
    print(softmax_outputs)
    print(softmax_outputs_diagflat)
    print(softmax_outputs_dot)
    print()
    print(softmax_outupts_sub)
