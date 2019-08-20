import numpy as np
import joblib
from matplotlib import pyplot as plt
from sklearn.metrics import r2_score as r2

# set paths
modelpath = './HLR/regression-model/'
inputpath = './HLR/input/'
outputpath = './HLR/output/'

# load hierarchical linkage histogram feature matrix
with open(inputpath + 'linkages.txt') as file:
        L = np.array([[float(digit) for digit in line.split()] for line in file])
# Load ground truth cluster number labels
cx = np.genfromtxt(inputpath + 'cx.txt')

# load trained regression model
regression_model = joblib.load(modelpath + 'regression_model.sav')

# get cluster number estimate
yhat_test = regression_model.predict(L)

# R2 score (coeff of correlation: -1 to 1)
score_train = r2(cx, yhat_test)
print('R^2: train=%.3f' % score_train)

# sort the outputs
yhat_sorted = list()
cindx = np.argsort(cx, axis=-1)
csorted = np.sort(cx, axis=-1)
for i in cindx:
    yhat_sorted.append(yhat_test[i])

samples = range(len(cindx))

# plot the results
plt.plot(samples, yhat_sorted, marker = 'o')
plt.plot(samples, csorted, 'r')
plt.show()

# save test results
np.savetxt(outputpath + 'output_test.txt', (yhat_sorted, csorted), fmt='%.3f')
