import numpy as np
import joblib
from matplotlib import pyplot as plt
from sklearn.metrics import r2_score as r2

## set paths
modelpath = './model/'
inputpath = './input/'
outputpath = './output/'

# load hierarchical linkage histogram feature matrix
with open(inputpath + 'linkages.txt') as file:
        L = np.array([[float(digit) for digit in line.split()] for line in file])
# Load ground truth cluster number labels
y = np.genfromtxt(inputpath + 'y.txt')

# load trained regression model
regression_model = joblib.load(modelpath + 'regression_model.sav')

# get cluster number estimate
yhat_test = regression_model.predict(L)

# R2 score (coeff of correlation: -1 to 1)
score_test = r2(y, yhat_test)
print('R^2: test=%.3f' % score_test)

# sort the outputs
yhat_sorted = list()
cindx = np.argsort(y, axis=-1)
csorted = np.sort(y, axis=-1)
for i in cindx:
    yhat_sorted.append(yhat_test[i])

samples = range(len(cindx))

# plot the results
plt.plot(samples, yhat_sorted, marker = 'o')
plt.plot(samples, csorted, 'r')
plt.text(5, 20, r'R$^2$='+str('%.3f' % score_test), fontsize=20)
plt.show()

# save test results
np.savetxt(outputpath + 'output_test.txt', (yhat_sorted, csorted), fmt='%.3f')
