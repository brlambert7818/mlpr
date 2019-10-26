import numpy as np

m = 2
sigma = 0.5
alpha = 4
n = 2

x1 = np.random.normal(m, sigma**2, 1000)
nu = np.random.normal(0, n**2, 1000)
x2 = alpha*x1 + nu 
print(np.cov(x1, x2))
print(16*2 + 4)


