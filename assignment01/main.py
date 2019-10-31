import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.optimize import minimize

### 1 ###

# load data
mat = sio.loadmat('amp_data.mat')
data = mat['amp_data']

## 1a ##

# plot sequence of data
# plt.plot(data)
# plt.xlabel('Time')
# plt.ylabel('Amplitude (m)')
# plt.show()
#
# # plot histogram of amplitudes
# plt.clf()
# plt.hist(data, bins=100)
# plt.xlabel('Amplitude (m)')
# plt.ylabel('Frequency')
# plt.show()

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
# print(dataTrain.shape[0] + dataVal.shape[0] + dataTest.shape[0]
# )

# separate into X and y matrices
X_shuf_train = dataTrain[:, :-1]
y_shuf_train = dataTrain[:, -1]
X_shuf_val = dataVal[:, :-1]
y_shuf_val = dataVal[:, -1]
X_shuf_test = dataTest[:, :-1]
y_shuf_test = dataTest[:, -1]
#
# ### 2 ###
#
# ## 2a ##
#
# ###plotting X and y train data from the first row
time_vector = np.arange(0, 20)[:, None]  # time vector for train data
# X_vector = X_shuf_train[0, :][:, None]  # training points
# y_pred = y_shuf_train[0]  # test point
# plt.clf()
# f3 = plt.figure()
# plt.plot(time_vector, X_vector, 'o', label='training points', markersize=10,
#          markeredgewidth=2) # plot X_train data
# plt.plot(20, y_pred, 'ro', label='test point', markersize=10, markeredgewidth=2) # plot y_train(prediction)
# plt.xlabel('Time (samples)', fontsize=14)
# plt.ylabel('Amplitude (m)', fontsize=14)
# #create and plot ticks for X and y train
# time_vector_all = np.vstack((time_vector,[20])) # vectors of all time, including for predcited from y_trai
# time_tick = [r'$\frac{'+str(id)+'}{20}$' for id in time_vector_all[:,-1]] # all time ticks
# plt.xticks(time_vector_all, time_tick)
# plt.tick_params(axis='both', which='major', labelsize=14)
# plt.show()
#
# ###fit a straight line
# #biased vector with 1's for linear fit
# def phi_linear(X):
#     return np.concatenate([np.ones((X.shape[0], 1)), X], axis=1)
#
# time_vector_biased = phi_linear(time_vector) #create a biased vector for linear fit
# w_fit_linear = np.linalg.lstsq(time_vector_biased, X_vector[:,-1], rcond=-1)[0] #fit
# f_grid_linear = np.dot(phi_linear(time_vector_all), w_fit_linear) # amplitudes predicted from straight line fit at all times
# plt.plot(time_vector_all,f_grid_linear,'-g', label='straight line fit', linewidth=3) #plot predicted line
#
#
# #time_vector_biased_short = phi_linear(time_vector[-5:,:])
# #w_fit_linear_short = np.linalg.lstsq(time_vector_biased_short, X_vector[-5:,-1], rcond=-1)[0] #fit
# #f_grid_linear_short = np.dot(phi_linear(time_vector[-5:,:]), w_fit_linear_short) # amplitudes predicted from straight line fit at all times
# #plt.plot(time_vector_all,f_grid_linear_short,'-k', label='straight line fit', linewidth=3) #plot predicted line
#
# ###fit a quartic line
# #feature vector for quartic fit
# def phi_quartic(X):
#     return np.concatenate( [np.ones((X.shape[0],1)), X, X**2, X**3, X**4], axis=1)
#
# quartic_feature_vector = phi_quartic(time_vector) #create a feature vector for a quartic fit
# w_fit_quartic = np.linalg.lstsq(quartic_feature_vector, X_vector[:,-1], rcond=-1)[0] #fit a quartic
# f_grid_quartic = np.dot(phi_quartic(time_vector_all), w_fit_quartic) # amplitudes predicted from quartic fit at all times
# plt.plot(time_vector_all,f_grid_quartic,'-k',label='quartic fit', linewidth=3) #plot predicted quartic curve
# plt.legend(loc='upper right')
# f3.savefig('2a_fits.pdf', bbox_inches='tight')
#
# ## 2c ##
#
# # Checking other snippets of length 20 - plot them
# randomRows = np.concatenate([[0],np.random.randint(0,X_shuf_train.shape[0],2) ])
# f4 = plt.gcf()
# for i in range(0,3): #plot on subplot each of three rows(random)
#     X_vector = X_shuf_train[randomRows[i],:][:,None] # training points
#     y_pred = y_shuf_train[randomRows[i]] # test point
#     plt.subplot(3,1,i+1)
#     plt.subplots_adjust(hspace = 0.9, left=0.2)
#     plt.title('Row %i' %randomRows[i], fontsize=16)
#     plt.xticks(time_vector_all, time_tick)
#     plt.plot(time_vector,X_vector,'o',label='training points', markersize=10,markeredgewidth=2) # plot X-train
#     plt.plot(20, y_pred,'ro', label='test point', markersize=10,markeredgewidth=2) # plot y_train(prediction)
#     axes = plt.gca()
#     plt.ylim(ymin=axes.get_ylim()[0]-0.14*np.abs(axes.get_ylim()[0]),ymax=axes.get_ylim()[1]+0.14*np.abs(axes.get_ylim()[1]))
#     plt.tick_params(axis='both', which='major', labelsize=16)
#     plt.xlabel('Time (samples)', fontsize=14)
#     plt.ylabel('Amplitude (m)', fontsize=14)
#     plt.legend(loc='upper right')
# plt.show()
# f4.set_size_inches(9, 7)
# f4.savefig('2c_randomplots.pdf')
#
# ys= np.random.randn(1000)
# xs = np.arange(0,1000)
# plt.plot(xs,ys)
# plt.show()

### 3 ###

## 3b ##

def Phi(C, K):
    phi = time_vector[-C:]**0
    for k in range(1, K):
        phi = np.concatenate((phi, (time_vector[-C:]/20)**k), axis=1)
    return phi

def makevv(C, K):
    phi = Phi(C, K)
    t1 = np.ones(K)
    return np.dot(np.dot(phi, np.linalg.inv(np.dot(phi.T, phi))), t1)

def pred_from_v(C, K, data):
    predv = np.dot(makevv(C, K).T, data[-C:, :])
    return predv

def mse(param,data):
    pred = pred_from_v(param[0], param[1], data).T[:, -1]
    MSE = np.square(pred - y_shuf_train).mean()
    return MSE

 ### 4 ###

 ## 4a ##

def findVC(data, ys):
    tempMSE = np.zeros((20, 1))
    V = []
    for i in range(1, 21):
        tempV = np.linalg.lstsq(data.T[-i:, :].T, ys[:, np.newanewaxis], rcond=-1)[0]
        V.append(tempV)
        tempMSE[i - 1] = np.square(np.dot(tempV.T, data.T[-i:, ]) - ys[:, np.newaxis].T).mean()
    optimC = np.argmin(tempMSE) + 1
    return optimC, V, tempMSE

def findVC_reg(data, ys):
    tempMSE = np.zeros((20, 1))
    V = []
    for i in range(1, 21):
        ys_new = ys[:, np.newaxis]
        y_reg = np.concatenate([ys_new, np.zeros((i, 1))])
        lam_matrix = np.sqrt(0.01) * np.identity(i)
        x_reg = np.concatenate([data.T[-i:, :].T, lam_matrix])
        tempV = np.linalg.lstsq(x_reg, y_reg, rcond=-1)[0]
        V.append(tempV)
        tempMSE[i - 1] = np.square(np.dot(tempV.T, x_reg.T) - y_reg.T).mean()
    optimC = np.argmin(tempMSE) + 1

    return optimC, V, tempMSE

def findVC_valid(data, ys, V):
    tempMSE = np.zeros((20, 1))
    for i in range(1, 21):
        tempMSE[i - 1] = np.square(np.dot(V[i-1].T, data.T[-i:, ]) - ys[:, np.newaxis].T).mean()
    optimC = np.argmin(tempMSE) + 1

    return optimC, V, tempMSE

# C, V, MSE = findVC(X_shuf_train, y_shuf_train)
# print(C)
# Cval, Vval, MSEvalid = findVC_valid(X_shuf_val, y_shuf_val, V)
# print(Cval)
# print(np.square(np.dot(V[-1].T, X_shuf_test.T[-20:, ]) - y_shuf_test[:, np.newaxis].T).mean())

# reg

# Cval, Vval, MSEvalid = findVC_valid(X_shuf_val, y_shuf_val, V)
# print(Cval)
# print(np.square(np.dot(V[-1].T, X_shuf_test.T[-20:, ]) - y_shuf_test[:, np.newaxis].T).mean())
y_pred = pred_from_v(2,2, X_shuf_val.T)
resid = y_pred - y_shuf_val
p = plt.figure()
plt.hist(np.log(resid+1), bins=1000)
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.show()