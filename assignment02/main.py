from ct_support_code import *
from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt

### Q1
data = loadmat('/Users/onyskj/ct_data.mat', squeeze_me=True)

X_train = data['X_train']
y_train = data['y_train']
X_val = data['X_val']
y_val = data['y_val']
X_test = data['X_test']
y_test = data['y_test']


## Q1a
#print(np.round(np.mean(y_train)) == 0)

# y_val mean and SE
mu_y_val = np.mean(y_val)
se_y_val = np.std(y_val) / np.sqrt(len(y_val))

# y_train mean and SE
mu_y_train = np.mean(y_train)
se_y_train = np.std(y_train) / np.sqrt(len(y_train))

# compare mean and SE between y_val and y_train
plt.errorbar(['y_train', 'y_val'], [mu_y_train, mu_y_val], yerr=[se_y_train, se_y_val],
             fmt='.k')
plt.show()

## Q1b

# find input features with constant values
X_train_T = X_train.T
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
n_features = len(X_train.T)
unique, uniq_index = np.unique(X_train, axis=1, return_index=True)
duplicate_indices = list(set(range(n_features)) - set(uniq_index))

# remove duplicate input features
for d in data_names:
     data[d] = np.delete(data[d], duplicate_indices, axis=1)

# check dimensions
# print('# final features = # original features - # constant features - # duplicate features')
# print(str(len(X_test.T)) + ' = ' + str(n_features_start) + ' - '
#       + str(len(const_cols)) + ' - ' + str(len(duplicate_indices)))


### Q2

# fit model with lstsq using fit_linreg()
w_fit = fit_linreg(X_train, y_train, 10)
y_pred_train = np.dot(phi_linear(X_train), w_fit)
y_pred_val = np.dot(phi_linear(X_val), w_fit)
print('full feature linreg rmse:')
print(rmse_lstsq(y_pred_train, y_train))
print(rmse_lstsq(y_pred_val, y_val))
print('')

 # fit model with gradient approach using fit_linreg_gradopt()
w_fitG, b_fitG = fit_linreg_gradopt(X_train, y_train, 10)
y_pred_train = np.add(np.dot(X_train, w_fitG), b_fitG)
y_pred_val = np.add(np.dot(X_val, w_fitG), b_fitG)
print('full feature gradopt rmse:')
print(rmse_grad(y_pred_train, y_train))
print(rmse_grad(y_pred_val, y_val))
print('')

# Yes, the results are the same, as the gradient descent optimizer approaches the LS-solution.


### Q3

## Q3a
D_test = X_test.shape[1]

# k = 10
R_10 = random_proj(D_test, 10)
X_train_reduc10 = np.matmul(X_train, R_10)
X_val_reduc10 = np.matmul(X_val, R_10)

w_fit_reduc10 = fit_linreg(X_train_reduc10, y_train, 10)
y_pred_train = np.dot(phi_linear(X_train_reduc10), w_fit_reduc10)
y_pred_val = np.dot(phi_linear(X_val_reduc10), w_fit_reduc10)
print('10 feature rmse:')
print(rmse_lstsq(y_pred_train, y_train))
print(rmse_lstsq(y_pred_val, y_val))
print('')

# k = 100
R_100 = random_proj(D_test, 100)
X_train_reduc100 = np.matmul(X_train, R_100)
X_val_reduc100 = np.matmul(X_val, R_100)

w_fit_reduc100 = fit_linreg(X_train_reduc100, y_train, 100)
y_pred_train = np.dot(phi_linear(X_train_reduc100), w_fit_reduc100)
y_pred_val = np.dot(phi_linear(X_val_reduc100), w_fit_reduc100)
print('100 feature rmse:')
print(rmse_lstsq(y_pred_train, y_train))
print(rmse_lstsq(y_pred_val, y_val))

# Q3b

# histogram of 46th feature and three extra random features for comparison
noBins = 15
whichFeatures = np.concatenate([[45],np.random.randint(0,X_train.shape[1],5) ])
f1 = plt.gcf()
for i in range(0,6):
    plt.subplot(2,3,i+1)
    plt.hist(X_train.T[whichFeatures[i]],bins=noBins)
    plt.title('hist. for feature '+ str(whichFeatures[i]+1))
    plt.xlabel('X_train values',fontsize=14)
    plt.ylabel('frequency',fontsize=14)
    plt.subplots_adjust(hspace = 0.8, left=0.1, wspace=0.5)
    plt.xticks(np.arange(-0.25,1.25,step=0.25))
plt.show()
f1.set_size_inches(12, 7)
f1.savefig('3b_hist.pdf')


no_of_val_train = X_train.shape[0]*X_train.shape[1] #number of all values in X_train
# X-train values = -0.25
count_25 = 0
for row in X_train:
    count_25 += (row == -0.25).sum()

frac_25 = count_25/(no_of_val_train)
print('Fraction of values=-0.25: ' + str(frac_25))

# X-train values = 0
count_0 = 0
for row in X_train:
    count_0 += (row == 0).sum()

frac_0 = count_0/(no_of_val_train)
print('Fraction of values=0: ' + str(frac_0))


# use aug_fn() to add extra binary features to X_train
X_train_aug = aug_fn(X_train)
X_val_aug = aug_fn(X_val)

# report rmse for augmented training and validation sets
w_fit = fit_linreg(X_train_aug, y_train, 10)
y_pred_train = np.dot(phi_linear(X_train_aug), w_fit)
y_pred_val = np.dot(phi_linear(X_val_aug), w_fit)
print('full feature(aug) linreg rmse:')
print(rmse_lstsq(y_pred_train, y_train))
print(rmse_lstsq(y_pred_val, y_val))


### Q4
K = 10  # number of thresholded classification problems to fit
mx = np.max(y_train)
mn = np.min(y_train)
hh = (mx-mn)/(K+1)
thresholds = np.linspace(mn+hh, mx-hh, num=K, endpoint=True)
Ws= np.zeros((X_train.shape[1]+1, K))

# Fit logistic regression with gradient descent optimiser
for kk in range(K):
    labels = y_train > thresholds[kk] # labels for train set

    # fit log reg for each class(this or other)
    w_fit_temp, b_fitG_temp = fit_logreg_gradopt(X_train, labels, 10)

    Ws[:-1,kk] = w_fit_temp
    Ws[-1,kk] = b_fitG_temp

#bias X's for sigmoid
X_bias_train = phi_linear(X_train)
X_bias_val = phi_linear(X_val)

#Predict probabilities using sigmoid
Pred_train = sigmoid(np.matmul(X_bias_train,Ws))
Pred_val = sigmoid(np.matmul(X_bias_val,Ws))

#Fit reg. linear regression using predictions from logreg to y
new_w_fit_train = fit_linreg(Pred_train, y_train, 10)
new_w_fit_val = fit_linreg(Pred_val, y_val, 10)

#Calculate RMSE's
y_pred_train = np.dot(phi_linear(Pred_train),new_w_fit_train)
y_pred_val = np.dot(phi_linear(Pred_val),new_w_fit_val)

print('RMSE for reg. linear regression on logreg predictions on train set: ')
print(rmse_lstsq(y_pred_train, y_train))
print('RMSE for reg. linear regression on logreg predictions on val set: ')
print(rmse_lstsq(y_pred_val, y_val))


