import numpy as np
import matplotlib.pyplot as plt

# original prior distribution of W: p(W) = N(W; O, 0.4^2)
# want to alter dist so that intercept varies more but dist of slopes is same


def sample_prior(n):
    W = np.random.normal(0, 0.5, (1000, 2)) 
    X = np.arange(-4, 4, .1)
    X = X[:, np.newaxis]
    X_bias = np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)
    Y = np.zeros((n,1))

    for i in range(n):
        W_sample = W[np.random.choice(W.shape[0], 1, replace=False), :]
        Y = np.dot(X_bias, W_sample.T)
        plt.plot(X, Y)  

    plt.show()


def posterior(N):
    m = np.random.normal(0, 1)
    y = np.random.normal(m, 1, N)
    y_bar = np.mean(y)
    post_sd = 1 / (1+N)
    post_mean = N*y_bar / (1 + N)
    return m, post_sd, post_mean


count_true_m = 0
for i in range(10000):
    m, post_sd, post_mean = posterior(12)
    if (m <= post_mean + post_sd) and (m >= post_mean - post_sd):
        count_true_m += 1

print(count_true_m / 10000)



