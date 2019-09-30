import numpy as np
import matplotlib.pyplot as plt

N = int(1e4)
D = 2
A = np.array([[1, 0], [0.5, 0.5]])
X = np.random.randn(N, D)
Z = A.dot(X.T)
Z = Z.T

plt.clf()
plt.plot(Z[:, 1], Z[:, 0], '.')
plt.axis('square')
plt.show()

# prove theoretical cov[x1, x2] where x1=x2 is roughly a square matrix of ones
# x1 = np.random.randn(50)
# cov = np.cov(x1, x1)

# def transformation(a):
#     x = np.random.randn(2)


