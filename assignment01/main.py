import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt

### 1 ###

# load data
mat = sio.loadmat('amp_data.mat')
data = mat['amp_data']

## 1a ##

# # plot sequence of data
# plt.plot(data)
# plt.xlabel('Time')
# plt.ylabel('Amplitude (m)')
# plt.show()
# """"
#
# """
#
# # plot histogram of amplitudes
# plt.clf()
# plt.hist(data, bins=100)
# plt.xlabel('Amplitude (m)')
# plt.ylabel('Frequency')
# plt.show()
# """"
#
# """

## 1b ##

# trim dataset to allow for reshape of CX21
n = len(data)
trimCount = n % 21
dataTrimmed = data[:-trimCount]
nTrim = len(dataTrimmed)

# reshape dataset to CX21 where C = number of times we can make a full
# row of 21 from the original dataset
nRows = int(nTrim/21)
dataReshaped = np.reshape(dataTrimmed, (nRows, 21))

# set random seed to 1234 for repeatability and then shuffle data
dataShuffled = np.random.RandomState(1234).permutation(dataReshaped)

# 70% train, 15% validation, 15% test
dataTrain = dataShuffled[:int(nRows*0.7), :]
dataVal = dataShuffled[int(nRows*0.7):int(nRows*0.85), :]
dataTest = dataShuffled[int(nRows*0.85):, :]

# check the num rows of the separated data sets add up to total rows in original
# print("nrows: ", nRows)
# print(dataTrain.shape[0] + dataVal.shape[0] + dataTest.shape[0])

# separate into X and y matrices
X_shuf_train = dataTrain[:, :-1]
y_shuf_train = dataTrain[:, -1]
X_shuf_val = dataVal[:, :-1]
y_shuf_val = dataVal[:, -1]
X_shuf_test = dataTest[:, :-1]
y_shuf_test = dataTest[:, -1]


