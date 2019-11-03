from ct_support_code import *
from scipy.io import loadmat
#import cupy as np
import numpy as np
#import cupy as cp
#import matplotlib.pyplot as plt
#from numba import vectorize


### Q1
#data = loadmat('/Users/onyskj/ct_data.mat', squeeze_me=True)
#@vectorize(['float32(float32, float32)'], target='cuda')
data = loadmat('data/ct_data.mat', squeeze_me=True)


X_train = data['X_train']
y_train = data['y_train']
X_val = data['X_val']
y_val = data['y_val']
X_test = data['X_test']
y_test = data['y_test']


## Q1a
#print(np.round(np.mean(y_train)) == 0)
#
## y_val mean and SE
#mu_y_val = np.mean(y_val)
#se_y_val = np.std(y_val) / np.sqrt(len(y_val))
#
## y_train mean and SE
#mu_y_train = np.mean(y_train)
#se_y_train = np.std(y_train) / np.sqrt(len(y_train))
#
## compare mean and SE between y_val and y_train
#plt.errorbar(['y_train', 'y_val'], [mu_y_train, mu_y_val], yerr=[se_y_train, se_y_val],
#             fmt='.k')
#plt.show()

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

## fit model with lstsq using fit_linreg()
#w_fit = fit_linreg(X_train, y_train, 10)
#y_pred_train = np.dot(phi_linear(X_train), w_fit)
#y_pred_val = np.dot(phi_linear(X_val), w_fit)
#print('full feature linreg rmse:')
#print(rmse_lstsq(y_pred_train, y_train))
#print(rmse_lstsq(y_pred_val, y_val))
#print('')
#
# # fit model with gradient approach using fit_linreg_gradopt()
#w_fitG, b_fitG = fit_linreg_gradopt(X_train, y_train, 10)
#y_pred_train = np.add(np.dot(X_train, w_fitG), b_fitG)
#y_pred_val = np.add(np.dot(X_val, w_fitG), b_fitG)
#print('full feature gradopt rmse:')
#print(rmse_grad(y_pred_train, y_train))
#print(rmse_grad(y_pred_val, y_val))
#print('')
#
## Yes, the results are the same, as the gradient descent optimizer approaches the LS-solution.
#
#
#### Q3
#
### Q3a
#D_test = X_test.shape[1]
#
## k = 10
#R_10 = random_proj(D_test, 10)
#X_train_reduc10 = np.matmul(X_train, R_10)
#X_val_reduc10 = np.matmul(X_val, R_10)
#
#w_fit_reduc10 = fit_linreg(X_train_reduc10, y_train, 10)
#y_pred_train = np.dot(phi_linear(X_train_reduc10), w_fit_reduc10)
#y_pred_val = np.dot(phi_linear(X_val_reduc10), w_fit_reduc10)
#print('10 feature rmse:')
#print(rmse_lstsq(y_pred_train, y_train))
#print(rmse_lstsq(y_pred_val, y_val))
#print('')
#
## k = 100
#R_100 = random_proj(D_test, 100)
#X_train_reduc100 = np.matmul(X_train, R_100)
#X_val_reduc100 = np.matmul(X_val, R_100)
#
#w_fit_reduc100 = fit_linreg(X_train_reduc100, y_train, 100)
#y_pred_train = np.dot(phi_linear(X_train_reduc100), w_fit_reduc100)
#y_pred_val = np.dot(phi_linear(X_val_reduc100), w_fit_reduc100)
#print('100 feature rmse:')
#print(rmse_lstsq(y_pred_train, y_train))
#print(rmse_lstsq(y_pred_val, y_val))
#
## Q3b
#
## histogram of 46th feature and three extra random features for comparison
#noBins = 15
#whichFeatures = np.concatenate([[45],np.random.randint(0,X_train.shape[1],5) ])
#f1 = plt.gcf()
#for i in range(0,6):
#    plt.subplot(2,3,i+1)
#    plt.hist(X_train.T[whichFeatures[i]],bins=noBins)
#    plt.title('hist. for feature '+ str(whichFeatures[i]+1))
#    plt.xlabel('X_train values',fontsize=14)
#    plt.ylabel('frequency',fontsize=14)
#    plt.subplots_adjust(hspace = 0.8, left=0.1, wspace=0.5)
#    plt.xticks(np.arange(-0.25,1.25,step=0.25))
#plt.show()
#f1.set_size_inches(12, 7)
#f1.savefig('3b_hist.pdf')
#
#
#no_of_val_train = X_train.shape[0]*X_train.shape[1] #number of all values in X_train
## X-train values = -0.25
#count_25 = 0
#for row in X_train:
#    count_25 += (row == -0.25).sum()
#
#frac_25 = count_25/(no_of_val_train)
#print('Fraction of values=-0.25: ' + str(frac_25))
#
## X-train values = 0
#count_0 = 0
#for row in X_train:
#    count_0 += (row == 0).sum()
#
#frac_0 = count_0/(no_of_val_train)
#print('Fraction of values=0: ' + str(frac_0))
#
#
## use aug_fn() to add extra binary features to X_train
#X_train_aug = aug_fn(X_train)
#X_val_aug = aug_fn(X_val)
#
## report rmse for augmented training and validation sets
#w_fit = fit_linreg(X_train_aug, y_train, 10)
#y_pred_train = np.dot(phi_linear(X_train_aug), w_fit)
#y_pred_val = np.dot(phi_linear(X_val_aug), w_fit)
#print('full feature(aug) linreg rmse:')
#print(rmse_lstsq(y_pred_train, y_train))
#print(rmse_lstsq(y_pred_val, y_val))
#
#
#### Q4
#D=X_train.shape[1]
#K = 10  # number of thresholded classification problems to fit
#mx = np.max(y_train)
#mn = np.min(y_train)
#hh = (mx-mn)/(K+1)
#thresholds = np.linspace(mn+hh, mx-hh, num=K, endpoint=True)
#Ws= np.zeros((X_train.shape[1]+1, K))
#
## Fit logistic regression with gradient descent optimiser
#for kk in range(K):
#    np.random.seed(1)
#    labels = y_train > thresholds[kk] # labels for train set
#
#    # fit log reg for each class(this or other)
#    init = (np.zeros(D), np.array(0)) #start with zeros
##    init = (np.random.randn(D),np.random.randn(1)) #start with random weights
#    w_fit_temp, b_fitG_temp = fit_logreg_gradopt(X_train, labels, 10, init)
#
#    Ws[:-1,kk] = w_fit_temp
#    Ws[-1,kk] = b_fitG_temp
#
##bias X's for sigmoid
#X_bias_train = phi_linear(X_train)
#X_bias_val = phi_linear(X_val)
#
##Predict probabilities using sigmoid
#Pred_train = sigmoid(np.matmul(X_bias_train,Ws))
#Pred_val = sigmoid(np.matmul(X_bias_val,Ws))
#
##Fit reg. linear regression using predictions from logreg to y
#new_w_fit = fit_linreg(Pred_train, y_train, 10)
#
##Calculate RMSE's
#y_pred_train = np.dot(phi_linear(Pred_train),new_w_fit)
#y_pred_val = np.dot(phi_linear(Pred_val),new_w_fit)
#
#print('RMSE for reg. linear regression on logreg predictions on train set: ')
#print(rmse_lstsq(y_pred_train, y_train))
#print('RMSE for reg. linear regression on logreg predictions on val set: ')
#print(rmse_lstsq(y_pred_val, y_val))
#
#
#### Q5
#scale_rand = 0.1/np.sqrt(K)
#init_ww = scale_rand*np.random.randn(K)
#init_bb = 0.1*np.random.randn(1)
#init_V = scale_rand*np.random.randn(K,D)
#init_bk = scale_rand*np.random.randn(K)
#init = (init_ww, init_bb, init_V, init_bk)
#
#ww_nn, bb_nn, V_nn, bk_nn  = fit_nn_gradopt(X_train, y_train, 10, init)
#
#a_train = np.dot(X_train, V_nn.T)+bk_nn
#P_train = sigmoid(a_train)
#y_pred_train = np.dot(P_train,ww_nn) + bb_nn
#print('RMSE for nn on train set: ')
#print(rmse_grad(y_pred_train, y_train))
#
#a_val = np.dot(X_val, V_nn.T)+bk_nn
#P_val = sigmoid(a_val)
#y_pred_val = np.dot(P_val,ww_nn) + bb_nn
#print('RMSE for nn on val set: ')
#print(rmse_grad(y_pred_val, y_val))
#
#
### Q5 B
#init_ww_B = new_w_fit[:-1,-1]
#init_bb_B = new_w_fit[-1,:]
#init_V_B = Ws[:-1,:].T
#init_bk_B = Ws[-1,:]
#init_B = (init_ww_B, init_bb_B, init_V_B, init_bk_B)
#
#ww_nn_B, bb_nn_B, V_nn_B, bk_nn_B  = fit_nn_gradopt(X_train, y_train, 10, init_B)
#
#a_train = np.dot(X_train, V_nn_B.T)+bk_nn_B
#P_train = sigmoid(a_train)
#y_pred_train = np.dot(P_train,ww_nn_B) + bb_nn_B
#print('RMSE for nn on train set: ')
#print(rmse_grad(y_pred_train, y_train))
#
#a_val = np.dot(X_val, V_nn_B.T)+bk_nn_B
#P_val = sigmoid(a_val)
#y_pred_val = np.dot(P_val,ww_nn_B) + bb_nn_B
#print('RMSE for nn on val set: ')
#print(rmse_grad(y_pred_val, y_val))

## Q6

K=10
D=X_train.shape[1]
scale_rand = 0.1/np.sqrt(K)
init_ww = scale_rand*np.random.randn(K)
init_bb = 0.1*np.random.randn(1)
init_V = scale_rand*np.random.randn(K,D)
init_bk = scale_rand*np.random.randn(K)

init_for_ep = np.abs(0.1*np.random.randn(1))
init_alpha = np.abs(0.1*np.random.randn(1)+10)
init = (init_ww, init_bb, init_V, init_bk)

#for_eps=np.concatenate([-np.logspace(-0.52,-0.4,8,endpoint=True),np.logspace(-4,2,8,endpoint=True)])
for_eps=np.logspace(-0.52,-0.4,10,endpoint=True)
alphas = np.logspace(-1.3,-0.824,10,endpoint=True)

params_pairs = np.zeros((len(for_eps)*len(alphas),2))
RMSE_train = np.zeros(len(params_pairs))
RMSE_val = np.zeros(len(params_pairs))
for i in range(len(for_eps)):
    for j in range(len(alphas)):
        params_pairs[j+i*len(for_eps),0]=for_eps[i]
        params_pairs[j+i*len(alphas),1]=alphas[j]
        
for i in range(len(params_pairs)):
    print(i)
    ww_nn, bb_nn, V_nn, bk_nn  = fit_nn_gradopt_eps(X_train, y_train, params_pairs[i,1], params_pairs[i,0], init)
    a_train = np.dot(X_train, V_nn.T)+bk_nn
    P_train = (1-sigmoid(params_pairs[i,0]))*sigmoid(a_train)+sigmoid(params_pairs[i,0])/2
    y_pred_train = np.dot(P_train,ww_nn) + bb_nn
    RMSE_train[i] = rmse_grad(y_pred_train, y_train)
    
    a_val = np.dot(X_val, V_nn.T)+bk_nn
    P_val = (1-sigmoid(params_pairs[i,0]))*sigmoid(a_val)+sigmoid(params_pairs[i,0])/2
    y_pred_val = np.dot(P_val,ww_nn) + bb_nn
    RMSE_val[i] = rmse_grad(y_pred_val, y_val)

print('on train set:')
print('rmse train: ')
print(RMSE_train.min())
print('params(ep, alpha) train: ')
print(params_pairs[RMSE_train.argmin()])

print('on val set:')
print('rmse val: ')
print(RMSE_val.min())
print('params(ep, alpha) val: ')
print(params_pairs[RMSE_val.argmin()])
    
#        ww_nn, bb_nn, V_nn, bk_nn  = fit_nn_gradopt_eps(X_train, y_train, for_eps[i], alphas[j], init)
#        
#        a_train = np.dot(X_train, V_nn.T)+bk_nn
#        P_train = (1-sigmoid(for_eps[i]))*sigmoid(a_train)+sigmoid(for_eps[i])/2
#        y_pred_train = np.dot(P_train,ww_nn) + bb_nn
#        rmse_train = rmse_grad(y_pred_train, y_train)
#        RMSE_train[i,j] = rmse_train
#        

        



#a_train = np.dot(X_train, V_nn.T)+bk_nn
#P_train = (1-sigmoid(for_ep_nn))*sigmoid(a_train)+sigmoid(for_ep_nn)/2
#y_pred_train = np.dot(P_train,ww_nn) + bb_nn
#print('RMSE for nn on train set: ')
#print(rmse_grad(y_pred_train, y_train))
#
#a_val = np.dot(X_val, V_nn.T)+bk_nn
#P_val = (1-sigmoid(for_ep_nn))*sigmoid(a_val)+sigmoid(for_ep_nn)/2
#y_pred_val = np.dot(P_val,ww_nn) + bb_nn
#print('RMSE for nn on val set: ')
#print(rmse_grad(y_pred_val, y_val))

#from ct_support_code import *
