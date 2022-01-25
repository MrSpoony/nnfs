import matplotlib.pyplot as plt
from nnfs.datasets import spiral_data
import nnfs

nnfs.init()
X, y = spiral_data(samples=1000, classes=3)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap="brg")

if __name__ == "__main__":
    plt.show()
