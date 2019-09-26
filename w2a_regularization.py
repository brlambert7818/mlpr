import numpy as np
import matplotlib.pyplot as plt


def rbf(x, c, h):
    return np.exp(-(x - c)**2 / h**2)


def generate_xy(N, D):
    mu = np.random.uniform(0, 8, N)
    mu_sin = np.sin(mu)
    print(mu_sin)
    x = np.tile(mu_sin[:, None], (1, D)) + 0.01 * np.random.randn(N, D)
    x_bias = np.concatenate([x, np.ones((N, 1))], axis=1)
    y = mu_sin[:, None] + 0.01 * np.random.randn(N, 1)
    w_fit = np.linalg.lstsq(rbf(x_bias, 1, 1), y)[0]

    print(np.dot(x_bias, w_fit).shape)
    plt.clf()
    plt.plot(x, y, 'b.')
    #plt.plot(x, np.dot(x_bias, w_fit), 'ro-')
    plt.show()


generate_xy(10, 5)















