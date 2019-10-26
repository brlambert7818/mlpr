import numpy as np
import matplotlib.pyplot as plt

"""
c1 = (0.5, 0.1, 0.2, 0.4, 0.3, 0.2, 0.2, 0.1, 0.35, 0.25)
c2 = (0.9, 0.8, 0.75, 1.0)

m1 = np.mean(c1)
sigma1 = np.var(c1)
m2 = np.mean(c2)
sigma2 = np.var(c2)
"""

# generate w_tilde
mu_w = np.array([-5, 0])
sigmaSq_w = 4
w1 = np.random.normal(mu_w[0], sigmaSq_w)
w2 = np.random.normal(mu_w[1], sigmaSq_w)
w_tilde = np.array([w1, w2])

# generate D1, D2
mu_x1 = 0
sigmaSq_x1 = 0.25
n_d1 = 15
x_d1 = np.random.normal(mu_x1, sigmaSq_x1, (n_d1, 2))

mu_y1 = np.dot(x_d1, w_tilde)
sigmaSq_y = 1
y_d1 = np.random.normal(mu_y1, sigmaSq_y, n_d1)

mu_x2 = 0.5
sigmaSq_x2 = 0.01
n_d2 = 30
x_d2 = np.random.normal(mu_x2, sigmaSq_x2, (n_d2, 2))

mu_y2 = np.dot(x_d2, w_tilde)
y_d2 = np.random.normal(mu_y2, sigmaSq_y, n_d2)

# visualize prior w_tilde
n_w = 400
W = np.zeros((n_w, 2))
for i in range(n_w):
    w1 = np.random.normal(mu_w[0], sigmaSq_w)
    w2 = np.random.normal(mu_w[1], sigmaSq_w)
    W[i, 0] = w1
    W[i, 1] = w2

plt.plot(W[:, 0], W[:, 1], linestyle='', marker='+', mew=2, ms=6)
plt.show()

# visualize posteriors

# p(w | D1)
# cov_w_d1 = sigmaSq_y*np.linalg.inv((sigmaSq_y*sigmaSq_w + np.dot(x_d1.T, x_d1)))
# mu_w_d1 = cov_w_d1*sigmaSq_w*mu_w + (1/sigmaSq_y)*np.dot(np.dot(cov_w_d1, x_d1.T), y_d1)

