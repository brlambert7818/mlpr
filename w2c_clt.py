import numpy as np
import matplotlib.pyplot as plt


def generate_nongaussian(K):
    return np.random.exponential(1, K)


def plot_distributions(K, N):
    sums = np.zeros(N)
    for i in range(N):
        x = generate_nongaussian(K)
        sum_x = np.sum(x)
        sums[i] = sum_x

    plt.clf()
    plt.hist(sums, bins=50)
    plt.show()

plot_distributions(1000, 10**6)






