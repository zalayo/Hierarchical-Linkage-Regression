import numpy as np
from matplotlib import pyplot as plt
from sklearn.neural_network import MLPRegressor as mlp
from sklearn.svm import SVR as svm
from sklearn.metrics import r2_score as r2
import pickle

# set paths
modelpath = './HLR/regression-model/'
inputpath = './HLR/input/'
outputpath = './HLR/output/'

### Feedforward Neural Network Regression Model
regression_model = mlp(hidden_layer_sizes=(100,50), activation='logistic', solver='adam', alpha=0.15, batch_size='auto',
                        learning_rate='adaptive', learning_rate_init=0.001, power_t=0.5, max_iter=1000, shuffle=True,
                        random_state=None, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True,
                        early_stopping=False, validation_fraction=0.2, beta_1=0.9, beta_2=0.999, epsilon=1e-08, n_iter_no_change=10)

### Support Vector Machine Regression Model
# regression_model = svm(kernel='rbf', C=1e6, gamma='auto', epsilon=0.1)

# load hierarchical linkage histogram feature matrix
with open(inputpath + 'linkages.txt') as file:
        L = np.array([[float(digit) for digit in line.split()] for line in file])
# load ground truth cluster number labels
cx = np.genfromtxt(inputpath + 'cx.txt')

# fit regression model
regression_model.fit(L, cx)

# estimate cluster number
yhat_train = regression_model.predict(L)
#yhat_test = regression_model.predict(Ltest)

# R2 score (coeff of correlation: -1 to 1)
score_train = r2(cx, yhat_train)
print('R^2: train=%.3f' % score_train)

# sort the outputs
yhat_sorted = list()
cindx = np.argsort(cx, axis=-1)
csorted = np.sort(cx, axis=-1)
for i in cindx:
    yhat_sorted.append(yhat_train[i])

samples = range(len(cindx))

# plot the results
plt.plot(samples, yhat_sorted, marker = 'o')
plt.plot(samples, csorted, 'r')
plt.show()

# save the model and the results of training
pickle.dump(regression_model, open(modelpath + 'regression_model.sav', 'wb'))
np.savetxt(outputpath + 'output_train.txt', (yhat_sorted, csorted), fmt='%.3f')
