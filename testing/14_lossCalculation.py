import math
import numpy as np

softmax_output = [0.7, 0.1, 0.2]

target_output = [1, 0, 0]

loss = -(math.log(softmax_output[0]) * target_output[0] +
         math.log(softmax_output[1]) * target_output[1] +
         math.log(softmax_output[2]) * target_output[2])
# Equal to
loss = -(math.log(softmax_output[0]))
# Because target_outupt is one at this position

if __name__ == '__main__':
    print(np.log())
    print(loss)