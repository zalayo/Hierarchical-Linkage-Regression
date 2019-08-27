
import numpy as np
from libHLR import randclust as rclust

## set paths
inputpath = './input/'

## Generate Random Clustered Data for training and testing
################################################################################
## Total number of clustering instances
K = 100
## maximum number of clusters allowed per clustering instance:
maxclust = 30
## embedding dimension of data (i.e. feature dimensionality)
dim = 50
## maximum span of cluster centers (applied to all axes):
# lower bound
lb = -10
# upper bound
ub = 10
## bounds on cluster standard deviation (applied to all axes):
# lower bound
sdl = 0.25
# upper bound
sdu = 2.5
## minimum number of samples allowed per cluster
minsamples = 10
## maximum number of samples allowed per cluster
maxsamples = 500
################################################################################

# seed random generator
np.random.seed()

# initialize HLR random cluster object
xRand = rclust(K)

# set user-defined parameters
xRand.maxclust = maxclust
xRand.dim = dim
xRand.lb = lb
xRand.ub = ub
xRand.sdl = sdl
xRand.sdu = sdu
xRand.minsamples = minsamples
xRand.maxsamples = maxsamples

# generate random clustered dataset
#   X = clustered data
#   y = ground truth cluster number
#   kx = clustering instance index
#   cx = cluster index which rolls over for each clustering instance
X, y, kx, cx = xRand.dataset()

# save generated data
np.savetxt(inputpath + 'X.txt', X, fmt='%.8f', delimiter='\t', newline='\n')
np.savetxt(inputpath + 'kx.txt', kx, fmt='%d')
np.savetxt(inputpath + 'cx.txt', cx, fmt='%d')
np.savetxt(inputpath + 'y.txt', y, fmt='%d')
