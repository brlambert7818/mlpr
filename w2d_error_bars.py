import numpy as np
import matplotlib.pyplot as plt

# Bernoulli dist
x = np.random.rand(1, 100) < 0.3
z = np.random.rand(1, 100) < 0.3

sigma_hat_x = np.std(x)
sigma_hat_z = np.std(z)
N = x.shape[1]

plt.clf()
plt.plot('x', np.mean(x), 'r.', markersize=12)
plt.plot('z', np.mean(z), 'r.', markersize=12)
plt.errorbar(['x', 'z'], [np.mean(x), np.mean(z)],
             [sigma_hat_x / np.sqrt(N), sigma_hat_x / np.sqrt(N)],
             fmt='none')
plt.show()
