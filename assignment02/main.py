#from assignment02.ct_support_code import *
from ct_support_code import *
from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
# import time


### Q1

data = loadmat('/Users/onyskj/ct_data.mat', squeeze_me=True)
#data = loadmat('/Users/brianlambert/Desktop/mlpr/assignment02/ct_data.mat', squeeze_me=True)
#data = loadmat('data/ct_data.mat', squeeze_me=True)

X_train = data['X_train']
y_train = data['y_train']
X_val = data['X_val']
y_val = data['y_val']
X_test = data['X_test']
y_test = data['y_test']

## Q1a

mu_y_train = np.mean(y_train) # y_train mean all positions
mu_y_val = np.mean(y_val) # y_val mean all positions
se_y_val = np.std(y_val) / np.sqrt(len(y_val)) #y_val se all positions

# y_train mean and SE for the first 5,785 positions
mu_y_train_5785 = np.mean(y_train[:5785])
se_y_train_5785 = np.std(y_train[:5785]) / np.sqrt(len(y_train[:5785]))

print('Mean of the training positions in y_train: ' + str(mu_y_train))
print('Mean of 5785 positions in y_val: ' + str(mu_y_val))
print('Standard error of 5785 positions in y_val: ' + str(se_y_val))
print('Mean of the first 5785 training positions in y_train: ' + str(mu_y_train_5785))
print('Standard error of the first 5785 training positions in y_train: ' + str(se_y_train_5785))

muS=np.zeros(5)
seS=np.zeros(5)
for i in range(5):
    muS[i] = np.mean(y_train[0+5785*i:5785*(i+1)])
    seS[i] = np.std(y_train[0+5785*i:5785*(i+1)]) / np.sqrt(len(y_train[0+5785*i:5785*(i+1)]))

# compare mean and SE between y_val and y_train
f1 = plt.gcf()
plt.plot(0.5, mu_y_train_5785, 'r.', markersize=15, label='y_train[:5785] mean')
plt.plot(1.5, mu_y_val, 'g.', markersize=15, label='y_val mean')
plt.errorbar(np.array([0.5,1.5]), [mu_y_train_5785, mu_y_val], yerr=[se_y_train_5785, se_y_val],
             fmt='none', elinewidth=3, capsize=6,ecolor='k', label='error bars')
#plt.plot([0, 2],[0, 0])
plt.legend()
plt.xlim(0,2)
plt.ylim()
plt.xticks(np.array([0.5,1.5]), ['y_train[:5785]','y_val'],fontsize=15)
plt.ylabel('values', fontsize=15)
#f1.set_size_inches(9, 6)
#f1.savefig('1a_error_bar_plot.pdf')
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

# remove input features with constant values
data_names = ['X_train', 'X_val', 'X_test']
for d in data_names:
    data[d] = np.delete(data[d], const_cols, axis=1)

print('Features with constant values: ' + str(const_cols))

# find duplicate input features
X_train_T = data['X_train'].T
n_features = len(X_train_T)
unique, uniq_index = np.unique(data['X_train'], axis=1, return_index=True)
duplicate_indices = np.sort(list(set(range(n_features)) - set(uniq_index)))

# remove duplicate input features
for d in data_names:
    data[d] = np.delete(data[d], duplicate_indices, axis=1)

print('Duplicate features: ' + str(duplicate_indices))

#reassign arrays
X_train = data['X_train']
y_train = data['y_train']
X_val = data['X_val']
y_val = data['y_val']
X_test = data['X_test']
y_test = data['y_test']

# check dimensions
# print('# final features = # original features - # constant features - # duplicate features')
# print(str(len(X_test.T)) + ' = ' + str(n_features_start) + ' - '
#       + str(len(const_cols)) + ' - ' + str(len(duplicate_indices)))


## Q2

# fit model with lstsq using fit_linreg()
w_fit = fit_linreg(X_train, y_train, 10)
y_pred_train = np.dot(phi_linear(X_train), w_fit)[:, -1]
y_pred_val = np.dot(phi_linear(X_val), w_fit)[:, -1]

# rmse from lstsq
rmse_train_lstsq = rmse(y_pred_train, y_train)
rmse_val_lstsq = rmse(y_pred_val, y_val)
print('full feature linreg rmse:')
print(rmse(y_pred_train, y_train))
print(rmse(y_pred_val, y_val))
print('')

# fit model with gradient approach using fit_linreg_gradopt()
w_fitG, b_fitG = fit_linreg_gradopt(X_train, y_train, 10)
y_pred_trainG = np.add(np.dot(X_train, w_fitG), b_fitG)
y_pred_valG = np.add(np.dot(X_val, w_fitG), b_fitG)

# rmse from gradopt
rmse_train_gradopt = rmse(y_pred_trainG, y_train)
rmse_val_gradopt = rmse(y_pred_valG, y_val)
print('full feature gradopt rmse:')
print(rmse(y_pred_trainG, y_train))
print(rmse(y_pred_valG, y_val))
print('')

# Yes, the results are the same, as the gradient descent optimizer approaches the LS-solution.


### Q3

## Q3a
D_test = X_test.shape[1] #number of features

# reduce X dimensions to k = 10
R_10 = random_proj(D_test, 10)
X_train_reduc10 = np.matmul(X_train, R_10)
X_val_reduc10 = np.matmul(X_val, R_10)

# fit linear regression model using k = 10 input dimensions
w_fit_reduc10 = fit_linreg(X_train_reduc10, y_train, 10)
y_pred_train10 = np.dot(phi_linear(X_train_reduc10), w_fit_reduc10)[:, -1]
y_pred_val10 = np.dot(phi_linear(X_val_reduc10), w_fit_reduc10)[:, -1]

#calcuate rmse for k=10
rmse_train_10 = rmse(y_pred_train10, y_train)
rmse_val_10 = rmse(y_pred_val10, y_val)
print('10 feature rmse:')
print(rmse(y_pred_train10, y_train))
print(rmse(y_pred_val10, y_val))
print('')


# reduce X dimensions to k = 100
R_100 = random_proj(D_test, 100)
X_train_reduc100 = np.matmul(X_train, R_100)
X_val_reduc100 = np.matmul(X_val, R_100)

# fit linear regression model using k = 10 input dimensions
w_fit_reduc100 = fit_linreg(X_train_reduc100, y_train, 100)
y_pred_train100 = np.dot(phi_linear(X_train_reduc100), w_fit_reduc100)[:, -1]
y_pred_val100 = np.dot(phi_linear(X_val_reduc100), w_fit_reduc100)[:, -1]

#calcuate rmse for k=100
rmse_train_100 = rmse(y_pred_train100, y_train)
rmse_val_100 = rmse(y_pred_val100, y_val)
print('100 feature rmse:')
print(rmse(y_pred_train100, y_train))
print(rmse(y_pred_val100, y_val))

# Q3b

# histogram of 46th feature and five extra random features for comparison
noBins = 15
whichFeatures = np.concatenate([[45], np.random.randint(0, X_train.shape[1], 5)])
f1 = plt.gcf()
for i in range(0, 6):
    plt.subplot(2, 3, i + 1)
    plt.hist(X_train.T[whichFeatures[i]], bins=noBins)
    plt.title('hist. for feature ' + str(whichFeatures[i] + 1))
    plt.xlabel('X_train values', fontsize=14)
    plt.ylabel('frequency', fontsize=14)
    plt.subplots_adjust(hspace=0.8, left=0.1, wspace=0.5)
    plt.xticks(np.arange(-0.25, 1.25, step=0.25))
plt.show()
#f1.set_size_inches(12, 7)
#f1.savefig('3b_hist.pdf')

no_of_val_train = X_train.shape[0] * X_train.shape[1]  # number of all values in X_train

# X-train values = -0.25
count_25 = 0
for row in X_train:
    count_25 += (row == -0.25).sum()

frac_25 = count_25 / no_of_val_train
print('Fraction of values=-0.25: ' + str(frac_25))

# X-train values = 0
count_0 = 0
for row in X_train:
    count_0 += (row == 0).sum()

frac_0 = count_0 / no_of_val_train
print('Fraction of values=0: ' + str(frac_0))

# use aug_fn() to add extra binary features to X_train
X_train_aug = aug_fn(X_train)
X_val_aug = aug_fn(X_val)

# report rmse for augmented training and validation sets
w_fit_aug = fit_linreg(X_train_aug, y_train, 10)
y_pred_train_aug = np.dot(phi_linear(X_train_aug), w_fit_aug)[:, -1]
y_pred_val_aug = np.dot(phi_linear(X_val_aug), w_fit_aug)[:, -1]

#Calculate RMSE for extra binary feature X
rmse_train_aug = rmse(y_pred_train_aug, y_train)
rmse_val_aug = rmse(y_pred_val_aug, y_val)

print('full feature(aug) linreg rmse:')
print(rmse(y_pred_train_aug, y_train))
print(rmse(y_pred_val_aug, y_val))


### Q4

D = X_train.shape[1]
K = 10 # number of thresholded classification problems to fit
mx = np.max(y_train)
mn = np.min(y_train)
hh = (mx - mn) / (K + 1)
thresholds = np.linspace(mn + hh, mx - hh, num=K, endpoint=True)

#matrix for storing weights for each class. task
Ws = np.zeros((X_train.shape[1] + 1, K)) 

# Fit logistic regression with gradient descent optimiser
for kk in range(K):
#    np.random.seed(1)
    labels = y_train > thresholds[kk]  # labels for train set

    # fit log reg for each class(this or other)
    init = (np.zeros(D), np.array(0))  # start with zeros
    w_fit_temp, b_fitG_temp = fit_logreg_gradopt(X_train, labels, 10, init)
    
    # store vectors w with bias b
    Ws[:-1, kk] = w_fit_temp
    Ws[-1, kk] = b_fitG_temp

# bias X's for sigmoid
X_bias_train = phi_linear(X_train)
X_bias_val = phi_linear(X_val)

# Predict probabilities using sigmoid
Pred_train = sigmoid(np.matmul(X_bias_train, Ws))
Pred_val = sigmoid(np.matmul(X_bias_val, Ws))


# Fit linear regression using predictions from logreg to y_train
new_w_fit = fit_linreg(Pred_train, y_train, 10)

# Calculate RMSE's
y_pred_train = np.dot(phi_linear(Pred_train), new_w_fit)[:, -1]
y_pred_val = np.dot(phi_linear(Pred_val), new_w_fit)[:, -1]
rmse_train_logreg_lin = rmse(y_pred_train, y_train)
rmse_val_logreg_lin = rmse(y_pred_val, y_val)

print('RMSE for reg. linear regression on logreg predictions on train set: ')
print(rmse(y_pred_train, y_train))
print('RMSE for reg. linear regression on logreg predictions on val set: ')
print(rmse(y_pred_val, y_val))

### Q5
# create initial randomized parameters for nn
scale_rand = 0.1 / np.sqrt(K)
init_ww = scale_rand * np.random.randn(K)
init_bb = 0.1 * np.random.randn(1)
init_V = scale_rand * np.random.randn(K, D)
init_bk = scale_rand * np.random.randn(K)
init = (init_ww, init_bb, init_V, init_bk)

# fit neural network with randomized initial parameters
ww_nn, bb_nn, V_nn, bk_nn = fit_nn_gradopt(X_train, y_train, 10, init)

# calculate nn rmse on training set
a_train = np.dot(X_train, V_nn.T) + bk_nn
P_train = sigmoid(a_train)
y_pred_train = np.dot(P_train, ww_nn) + bb_nn
print('RMSE for nn on train set: ')
print(rmse(y_pred_train, y_train))

# calculate nn rmse on validation set
a_val = np.dot(X_val, V_nn.T) + bk_nn
P_val = sigmoid(a_val)
y_pred_val = np.dot(P_val, ww_nn) + bb_nn
print('RMSE for nn on val set: ')
print(rmse(y_pred_val, y_val))
## Q5 B

# create initial parameters for nn using fits from Q4
init_ww_B = new_w_fit[:-1, -1]
init_bb_B = new_w_fit[-1, :]
init_V_B = Ws[:-1, :].T
init_bk_B = Ws[-1, :]
init_B = (init_ww_B, init_bb_B, init_V_B, init_bk_B)

# fit neural network with fitted parameters from Q4
ww_nn_B, bb_nn_B, V_nn_B, bk_nn_B = fit_nn_gradopt(X_train, y_train, 10, init_B)

# calculate nn rmse on training set
a_train = np.dot(X_train, V_nn_B.T) + bk_nn_B
P_train = sigmoid(a_train)
y_pred_train = np.dot(P_train, ww_nn_B) + bb_nn_B
print('RMSE for nn on train set: ')
print(rmse(y_pred_train, y_train))

# calculate nn rmse on validation set
a_val = np.dot(X_val, V_nn_B.T) + bk_nn_B
P_val = sigmoid(a_val)
y_pred_val = np.dot(P_val, ww_nn_B) + bb_nn_B
print('RMSE for nn on val set: ')
print(rmse(y_pred_val, y_val))

from ct_support_code import *



