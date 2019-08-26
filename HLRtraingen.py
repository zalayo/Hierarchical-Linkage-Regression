import sys
import numpy as np
from matplotlib import pyplot as plt
from sklearn.neural_network import MLPRegressor as mlp
from sklearn.svm import SVR as svm
from sklearn.metrics import r2_score as r2
from libHLR import randclust as rclust
from libHLR import linkages as link
import pickle

## select regression model ('mlp' or 'svm')
modeltype = 'svm'

## set paths
modelpath = './model/'
inputpath = './input/'
outputpath = './output/'

## set linkage parameters
# linkage method
linktype = 'complete'
# distance metric
distance = 'manhattan'
# bin count for 2D histogram
R = 40

## set total number of clustering instances to generate
K = 100

def init_model(modeltype):
    if modeltype == 'mlp':
        ### Feedforward Neural Network Regression Model
        regression_model = mlp(hidden_layer_sizes=(100,50),
                                activation = 'logistic',
                                solver = 'adam',
                                alpha = 0.2,
                                batch_size = 'auto',
                                learning_rate = 'adaptive',
                                learning_rate_init = 0.001,
                                power_t = 0.5,
                                max_iter = 1000,
                                shuffle = True,
                                random_state = None,
                                tol = 0.0001,
                                verbose = False,
                                warm_start = False,
                                momentum = 0.9,
                                nesterovs_momentum = True,
                                early_stopping = False,
                                validation_fraction = 0.1,
                                beta_1 = 0.9,
                                beta_2 = 0.999,
                                epsilon = 1e-08,
                                n_iter_no_change = 10)
    elif modeltype == 'svm':
        ### Support Vector Machine Regression Model
        regression_model = svm(kernel='rbf',
                                C=1e6,
                                epsilon=0.1,
                                gamma='auto',
                                tol=0.001,
                                cache_size=2000,
                                shrinking=True,
                                verbose=False,
                                max_iter=-1)
    return regression_model

def compute_linkages(X, kx):
    # initialize
    L = np.asarray([])
    idx = list()
    # Get number of clustering instances, K, in the data set
    K = xRand.K
    # extract Linkages from input
    for j in range(K):
        print('Processing linkage %d' % (j + 1), ' of %d' % K)
        idx = np.where(kx == j + 1)
        Xi = X[idx, :]
        Xi = Xi[0, :, :]
        L = np.concatenate((L, xLink.get(Xi)))
    L = np.hsplit(L, K)
    return L

### MAIN PROGRAM ###

# seed random generator
np.random.seed()

# initialize HLR class objects
xLink = link(distance, linktype)
xLink.R = R
xRand = rclust(K)

print('Generating randomly clustered dataset...\n')
# generate random clustered dataset
#   X = clustered data
#   y = ground truth cluster number
#   kx = clustering instance index
#   cx = cluster index which rolls over for each clustering instance
X, y, kx, cx = xRand.dataset()

# get linkage feature matrix
L = compute_linkages(X, kx)

# initialize regression model
regression_model = init_model(modeltype)

# fit regression model
regression_model.fit(L, y)

# estimate cluster number
yhat_train = regression_model.predict(L)
#yhat_test = regression_model.predict(Ltest)

# sort the outputs
yhat_sorted = list()
cindx = np.argsort(y, axis=-1)
csorted = np.sort(y, axis=-1)
for i in cindx:
    yhat_sorted.append(yhat_train[i])

# save the model, data, linkages and the results of training
pickle.dump(regression_model, open(modelpath + 'regression_model.sav', 'wb'))
np.savetxt(outputpath + 'output_train.txt', (yhat_sorted, csorted), fmt='%.3f')
np.savetxt(inputpath + 'linkages.txt', L, fmt='%.8f',delimiter='\t',newline='\n')
np.savetxt(inputpath + 'X.txt', X, fmt='%.8f',delimiter='\t',newline='\n')
np.savetxt(inputpath + 'kx.txt', kx, fmt='%d')
np.savetxt(inputpath + 'cx.txt', cx, fmt='%d')
np.savetxt(inputpath + 'y.txt', y, fmt='%d')

# R2 score (coeff of correlation: -1 to 1)
score_train = r2(y, yhat_train)
print('R^2: train=%.3f' % score_train)

# plot the results
samples = range(len(cindx))
plt.plot(samples, yhat_sorted, marker = 'o')
plt.plot(samples, csorted, 'r')
plt.text(5, 20, r'R$^2$='+str('%.3f' % score_train), fontsize=20)
plt.show()
