import numpy as np
import matplotlib.pyplot as plt


def average_noise(N, D, lam):
    mu = np.random.uniform(0, 1, N)
    x = np.tile(mu[:, None], (1, D)) + 0.01*np.random.randn(N, D)
    y = mu[:, None] + 0.01*np.random.randn(N, 1)

    y_reg = np.concatenate([y, np.zeros((D+1, 1))])
    lam_matrix = np.sqrt(lam)*np.identity(D+1)
    x_bias = np.concatenate([x, np.ones((x.shape[0], 1))], axis=1)
    x_reg = np.concatenate([x_bias, lam_matrix])
    w_fit = np.linalg.lstsq(x_bias, y)[0]
    w_fit_reg = np.linalg.lstsq(x_reg, y_reg)[0]
    print(y_reg.shape)
    print(x_bias.shape)
    print(x_reg.shape)
    print(w_fit_reg.shape)

    # plt.clf()
    # plt.plot(x, y, 'b.')
    # plt.plot(x, np.dot(x_bias, w_fit), 'r.')
    # plt.plot(x, x_bias, w_fit_reg, 'g.')
    # plt.show()


average_noise(50, 3, 3)














