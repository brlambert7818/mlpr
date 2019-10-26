#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 16:19:16 2019

@author: onyskj lambertb

"""

import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from fractions import Fraction

### 1 ###

# load data
mat = sio.loadmat('amp_data.mat')
data = mat['amp_data']

## 1a ##

# plot sequence of data
f1 = plt.figure()
plt.plot(data, linewidth=1)
plt.xlabel('Time (samples)')
plt.ylabel('Amplitude (m)')
plt.show()
f1.savefig('1a_linegraph.pdf', bbox_inches='tight')

# plot histogram of amplitudes
plt.clf()
f2 = plt.figure()
plt.hist(data, bins=1000)
plt.xlabel('Amplitude (m)')
plt.ylabel('Frequency')
plt.show()
f2.savefig('1a_histogram.pdf', bbox_inches='tight')

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
# or just  dataReshaped = np.reshape(dataTrimmed, (-1, 21))

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



### 2 ###

#plotting X and y train data from the first row
time_vector = np.arange(0,20)[:,None] # time vector for train data
X_vector = X_shuf_train[0,:][:,None]
y_pred = y_shuf_train[0]
plt.plot(time_vector,X_vector,'o',label='X train data', markersize=10,markeredgewidth=2) # plot X_train data
plt.plot(20, y_pred,'ro', label='y train data point', markersize=10,markeredgewidth=2) # plot y_train(prediction)
plt.xlabel('Time (samples)', fontsize=14)
plt.ylabel('Amplitude (m)', fontsize=14)
#create and plot ticks for X and y train
time_vector_all = np.vstack((time_vector,[20])) # vectors of all time, including for predcited from y_trai
time_tick = [r'$\frac{'+str(id)+'}{20}$' for id in time_vector_all[:,-1]] # all time ticks
plt.xticks(time_vector_all, time_tick)
plt.tick_params(axis='both', which='major', labelsize=14)
plt.show()


#fit a straight line

#biased vector with 1's for linear fit
def phi_linear(X):
    return np.concatenate( [np.ones((X.shape[0],1)), X], axis=1)

time_vector_biased = phi_linear(time_vector) #create a biased vector for linear fit
w_fit_linear = np.linalg.lstsq(time_vector_biased, X_vector[:,-1], rcond=-1)[0] #fit
f_grid_linear = np.dot(phi_linear(time_vector_all), w_fit_linear) # amplitudes predicted from straight line fit at all times
plt.plot(time_vector_all,f_grid_linear,'-g', label='linear fit', linewidth=3) #plot predicted line

#fit a quartic line

#feature vector for quartic fit
def phi_quartic(X):
    return np.concatenate( [np.ones((X.shape[0],1)), X, X**2, X**3, X**4], axis=1)

quartic_feature_vector = phi_quartic(time_vector) #create a feature vector for a quartic fit
w_fit_quartic = np.linalg.lstsq(quartic_feature_vector, X_vector[:,-1], rcond=-1)[0] #fit a quartic
f_grid_quartic = np.dot(phi_quartic(time_vector_all), w_fit_quartic) # amplitudes predicted from quartic fit at all times
plt.plot(time_vector_all,f_grid_quartic,'-k',label='quartic fit', linewidth=3) #plot predicted quartic curve

plt.legend(loc='upper right')
