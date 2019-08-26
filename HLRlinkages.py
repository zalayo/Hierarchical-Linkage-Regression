import os
import numpy as np
from sklearn.cluster import AgglomerativeClustering as agglom
from libHLR import linkages as link

## set paths
inputpath = './input/'

## set linkage parameters
# linkage method
linktype = 'complete'
# distance metric
distance = 'manhattan'
# bin count for 2D histogram
R = 40

# initialize HLR class object
xLink = link(distance, linktype)
xLink.R = R

#import data set
with open(inputpath + 'X.txt') as file:
    X = np.array([[float(digit) for digit in line.split()] for line in file])
# import clustering instance labels but check to see if it exists first
if not os.path.exists(inputpath + 'kx.txt'):
    kx = np.ones((np.shape(X)[0],))
else:
    kx = np.genfromtxt(inputpath + 'kx.txt')

# figure out number of samples in the data set
lastindx = len(kx) - 2
K = len(np.unique(kx[0:lastindx]))
idx = 0

# extract Linkages from input
L = np.asarray([])
for j in range(K):
    print('Processing linkage %d' % (j + 1), ' of %d' % K)
    idx = np.where(kx[0:lastindx] == j + 1)
    Xi = X[idx, :]
    Xi = Xi[0, :, :]
    L = np.concatenate((L, xLink.get(Xi)))
L = np.hsplit(L, K)

# write to file
np.savetxt(inputpath + 'linkages.txt', L, fmt='%.8f',
                                delimiter='\t',newline='\n')
