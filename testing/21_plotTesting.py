import numpy as np
from matplotlib import pyplot as plt


def f(x):
    return 2 * x ** 2


x = np.array(np.arange(0, 5, 0.001))
y = f(x)

print(x)
print(y)

if __name__ == '__main__':
    plt.plot(x, y)
    plt.show()
