import numpy as np
from matplotlib import pyplot as plt


def f(x):
    return 2 * x ** 3

def fPrime(x):
    return 6 * x ** 2

def approximateTangentLine(x, derivative, b):
    return derivative * x + b


delta = 0.0001
rangeToPlot = 10
handleSize = 1.8/2

x = np.array(np.arange(0, rangeToPlot, 0.001))
y = f(x)
plt.plot(x, y)

xPrime = np.array(np.arange(0, rangeToPlot, 0.001))
yPrime = fPrime(xPrime)
plt.plot(xPrime, yPrime)

for i in range(rangeToPlot):
    x1 = i
    x2 = i + delta

    y1 = f(x1)
    y2 = f(x2)

    approximateDerivative = (y2 - y1) / (x2 - x1)
    b = y2 - approximateDerivative * x2

    rangeToPlotDerivative = [x1 - handleSize, x1, x1 + handleSize]
    plt.plot(rangeToPlotDerivative,
             [approximateTangentLine(point, approximateDerivative, b) for point in rangeToPlotDerivative])

if __name__ == '__main__':
    plt.show()
