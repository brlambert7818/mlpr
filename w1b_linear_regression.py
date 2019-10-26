import numpy as np
import matplotlib.pyplot as plt


def rbf_1d(xx, cc1, cc2, hh):
    return 2*np.exp(-(xx-cc1)**2 / hh**2) - np.exp(-(xx-cc2)**2 / hh**2)


def sigmoid(xx, vv, bb):
    return 1/(1 + np.exp(-vv*xx - bb))


# grid_size = 0.1
# x_grid = np.arange(-10, 10, grid_size)
# plt.clf()
# plt.plot(x_grid, rbf_1d(x_grid, cc1=-5, cc2=5, hh=1), '-b')
# plt.show()

yy = np.array([1.1, 2.3, 2.9])
xx = np.array([0.8, 1.9, 3.1])

# plt.clf()
# plt.plot(xx, yy, '-b')
# plt.show()


def phi_linear(xin):
    return np.concatenate([xin, np.ones((xin.shape[0], 1))], axis=1)  # (N,D+1)


def phi_quadratic(xin):
    return np.concatenate([xin, xin**2, np.ones((xin.shape[0], 1))], axis=1)  # (N,D+2)


def fw_rbf(xx, cc):
    return np.exp(-(xx-cc)**2 / 2)


def phi_rbf(xin):
    return np.array([fw_rbf(xin, 1), fw_rbf(xin, 2), fw_rbf(xin, 3)])


def fit_and_plot(phi_fn, xx, yy):
    w_fit = np.linalg.lstsq(phi_fn(xx), yy)[0]
    x_grid = np.arange(0, 4, 0.1)
    y_grid = phi_fn(x_grid)*w_fit
    plt.clf()
    plt.plot(x_grid, y_grid, '-b')
    plt.show()


#fit_and_plot(phi_linear, xx, yy)
x = np.random.normal(0, 1, (5, 20))
phi_x = phi_quadratic(x)