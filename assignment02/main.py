from assignment02.ct_support_code import *
from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
#adasdas
### Q1
data = loadmat('ct_data.mat', squeeze_me=True)

## Q1a
#print(np.round(np.mean(data['y_train'])) == 0)

# y_val mean and SE
mu_y_val = np.mean(data['y_val'])
se_y_val = np.std(data['y_val']) / np.sqrt(len(data['y_val']))

# y_train mean and SE
mu_y_train = np.mean(data['y_train'])
se_y_train = np.std(data['y_train']) / np.sqrt(len(data['y_train']))

# compare mean and SE between y_val and y_train
plt.errorbar(['y_train', 'y_val'], [mu_y_train, mu_y_val], yerr=[se_y_train, se_y_val],
             fmt='.k')
plt.show()

## Q1b

# find input features with constant values
X_train_T = data['X_train'].T
n_features_start = len(X_train_T)
const_cols = []
i = 0
for col in X_train_T:
    if len(set(col)) == 1:
        const_cols.append(i)
    i += 1
# print(non_uniq_cols)

# remove input features with constant values
data_names = ['X_train', 'X_val', 'X_test']
for d in data_names:
    data[d] = np.delete(data[d], const_cols, axis=1)

# find duplicate input features
n_features = len(data['X_train'].T)
unique, uniq_index = np.unique(data['X_train'], axis=1, return_index=True)
duplicate_indices = list(set(range(n_features)) - set(uniq_index))

# remove duplicate input features
# for d in data_names:
#     data[d] = np.delete(data[d], duplicate_indices, axis=1)

# check dimensions
# print('# final features = # original features - # constant features - # duplicate features')
# print(str(len(data['X_test'].T)) + ' = ' + str(n_features_start) + ' - '
#       + str(len(const_cols)) + ' - ' + str(len(duplicate_indices)))


### Q2

# fit model with lstsq using fit_linreg()
# w_fit = fit_linreg(data['X_train'], data['y_train'], 10)
# print('full feature linreg rmse:')
# print(rmse_lstsq(w_fit, data['X_train'], data['y_train']))
# print(rmse_lstsq(w_fit, data['X_val'], data['y_val']))
# print('')
#
# # fit model with gradient approach using fit_linreg_gradopt()
# w_fit, b_fit = fit_linreg_gradopt(data['X_train'], data['y_train'], 10)
# print('full feature gradopt rmse:')
# print(rmse_grad(w_fit, b_fit, data['X_train'], data['y_train']))
# print(rmse_grad(w_fit, b_fit, data['X_val'], data['y_val']))
# print('')
#
# ### Q3
#
# ## Q3a
# D_test = data['X_test'].shape[1]
#
# # k = 10
# R_10 = random_proj(D_test, 10)
# X_train_reduc = np.matmul(data['X_train'], R_10)
# X_val_reduc = np.matmul(data['X_val'], R_10)
#
# w_fit = fit_linreg(X_train_reduc, data['y_train'], 10)
# print('10 feature rmse:')
# print(rmse_lstsq(w_fit, X_train_reduc, data['y_train']))
# print(rmse_lstsq(w_fit, X_val_reduc, data['y_val']))
# print('')
#
# # k = 100
# R_100 = random_proj(D_test, 100)
# X_train_reduc = np.matmul(data['X_train'], R_100)
# X_val_reduc = np.matmul(data['X_val'], R_100)
#
# w_fit = fit_linreg(X_train_reduc, data['y_train'], 100)
# print('100 feature rmse:')
# print(rmse_lstsq(w_fit, X_train_reduc, data['y_train']))
# print(rmse_lstsq(w_fit, X_val_reduc, data['y_val']))

## Q3b

# histogram of 46th feature
# plt.hist(data['X_train'].T[45])
# plt.show()

# % X-train values = -0.25
# count_25 = 0
# for row in data['X_train']:
#     count_25 += (row == -0.25).sum()

# % X-train values = 0
# count_0 = 0
# for row in data['X_train']:
#     count_0 += (row == 0).sum()

# use aug_fn() to add extra binary features to X_train
X_train_aug = aug_fn(data['X_train'])
X_val_aug = aug_fn(data['X_val'])

# report rmse for augmented training and validation sets
w_fit = fit_linreg(X_train_aug, data['y_train'], 10)
print('full feature linreg rmse:')
print(rmse_lstsq(w_fit, X_train_aug, data['y_train']))
print(rmse_lstsq(w_fit, X_val_aug, data['y_val']))
