import numpy as np
import joblib

# set paths
modelpath = './HLR/regression-model/'
inputpath = './HLR/input/'
outputpath = './HLR/output/'

# load hierarchical linkage histogram feature matrix
with open(inputpath + 'linkages.txt') as file:
        L = np.array([[float(digit) for digit in line.split()] for line in file])

# load regression model
regression_model = joblib.load(modelpath + 'regression_model.sav')

# predict
yhat = regression_model.predict(L)
#yhat_test = regression_model.predict(Ltest)

np.savetxt(outputpath + 'output.txt', yhat, fmt='%.3f')
