import math

import numpy as np

softmax_output = [0.7, 0.1, 0.2]

target_output = [1, 0, 0]

loss = -(math.log(softmax_output[0]) * target_output[0] +
         math.log(softmax_output[1]) * target_output[1] +
         math.log(softmax_output[2]) * target_output[2])
# Equal to
same_loss = -(math.log(softmax_output[0]))
# Because target_output is one at this position

softmax_outputs = [[0.7, 0.1, 0.2],
                   [0.1, 0.5, 0.4],
                   [0.02, 0.9, 0.08]]
# With indices
class_targets = [0, 1, 1]

class_loss = []
for target_index, distribution in zip(class_targets, softmax_outputs):
    class_loss.append(-np.log(distribution[target_index]))

softmax_outputs_as_np_array = np.array([[0.7, 0.1, 0.2],
                                        [0.1, 0.5, 0.4],
                                        [0.02, 0.9, 0.08]])
final_loss_calculation = -np.log(softmax_outputs_as_np_array[[range(len(softmax_outputs_as_np_array))], class_targets])
average_loss = np.mean(final_loss_calculation)

if __name__ == '__main__':
    print(loss)
    print(same_loss)
    print(class_loss)
    print(final_loss_calculation)
    print(average_loss)
