import numpy as np
import matplotlib.pyplot as plt


x = np.random.randn(10**6)
mean_x = np.mean(x)
var_x = np.var(x)


def pdf(x_input):
    return 1 / (np.sqrt(2*np.pi)) * np.exp((-1/2) * x_input**2)


hist = plt.hist(x, bins=100)
bin_centers = 0.5*(hist[1][1:] + hist[1][:-1])
bin_width = bin_centers[1] - bin_centers[0]
predicted_bin_heights = pdf(bin_centers)*10**6*bin_width

plt.clf()
plt.hist(x, bins=100)
plt.plot(bin_centers, predicted_bin_heights, '-r')
plt.show()

#this is a test

