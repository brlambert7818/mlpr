from assignment02.ct_support_code import *
from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt


### Q1
data = loadmat('ct_data.mat')

## Q1a
#print(np.round(np.mean(data['y_train'])) == 0)

# y_val mean and SE
mu_y_val = np.mean(data['y_val'])
se_y_val = np.std(data['y_val']) / np.sqrt(len(data['y_val']))

# y_train mean and SE
mu_y_train = np.mean(data['y_train'])
se_y_train = np.std(data['y_train']) / np.sqrt(len(data['y_train']))

# compare mean and SE between y_val and y_train
# plt.errorbar(['y_train', 'y_val'], [mu_y_train, mu_y_val], yerr=[se_y_train, se_y_val],
#              fmt='.k')
# plt.show()

## Q1b

# find input features with constant values
X_test_T = data['X_test'].T
n_features_start = len(X_test_T)
const_cols = []
i = 0
for col in X_test_T:
    if len(set(col)) == 1:
        const_cols.append(i)
    i += 1
# print(non_uniq_cols)

# remove input features with constant values
data_names = ['X_train', 'X_val', 'X_test']
for d in data_names:
    data[d] = np.delete(data[d], const_cols, axis=1)

# find duplicate input features
n_features = len(data['X_test'].T)
unique, uniq_index = np.unique(data['X_test'], axis=1, return_index=True)
duplicate_indices = list(set(range(n_features)) - set(uniq_index))

# remove duplicate input features
for d in data_names:
    data[d] = np.delete(data[d], duplicate_indices, axis=1)

# check dimensions
# print('# final features = # original features - # constant features - # duplicate features')
# print(str(len(data['X_test'].T)) + ' = ' + str(n_features_start) + ' - '
#       + str(len(const_cols)) + ' - ' + str(len(duplicate_indices)))


### Q2

# fit model with lstsq using fit_linreg()
w_fit = fit_linreg(data['X_train'], data['y_train'], 10)
print(w_fit.shape)
print(rmse_lstsq(w_fit, data['X_train'], data['y_train']))
print(rmse_lstsq(w_fit, data['X_val'], data['y_val']))

# fit model with gradient approach using fit_linreg_gradopt()
# w_fit, b_fit = fit_linreg_gradopt(data['X_train'], data['y_train'], 10)
# print(rmse_grad(w_fit, b_fit, data['X_train'], data['y_train']))
# print(rmse_grad(w_fit, b_fit, data['X_val'], data['y_val']))
