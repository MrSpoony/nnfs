import numpy as np

from common.ActivationSoftmaxLossCategoricalCrossentropy import ActivationSoftmaxLossCategoricalCrossentropy
from common.ActivationSoftmax import ActivationSoftmax
from common.LossCategoricalCrossEntropy import LossCategoricalCrossEntropy


softmaxOutputs = np.array([[0.7, 0.1, 0.2],
                            [0.1, 0.5, 0.4],
                            [0.02, 0.9, 0.08]])
classTargets = np.array([0, 1, 1])

softmaxLoss = ActivationSoftmaxLossCategoricalCrossentropy()
softmaxLoss.backward(softmaxOutputs, classTargets)
dvalues1 = softmaxLoss.dinputs

activation = ActivationSoftmax()
activation.output = softmaxOutputs
loss = LossCategoricalCrossEntropy()
loss.backward(softmaxOutputs, classTargets)
activation.backward(loss.dinputs)
