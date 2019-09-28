import numpy as np
import matplotlib.pyplot as plt

N = int(1e4)
D = 2

X = np.random.randn(N, D)
plt.clf()
plt.plot(X[:, 1], X[:, 0], '.')
plt.axis('square')
plt.show()
