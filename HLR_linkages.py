import numpy as np
from sklearn.cluster import AgglomerativeClustering as agglom

def get_linkages(X, linkage):
    # bin count for 2d histogram (one side)
    R = 40
    eps = np.finfo(np.float).eps

    # Number of samples in X
    sample_num = np.shape(X)[0]

    # Generate linkage hierarchy
    linkage.fit(X)

    # Get internal node coordinates of linkage tree
    Z = np.asarray(linkage.children_ + 1)
    Z.reshape((linkage.n_leaves_ - 1, 2))
    # print(Z[:,0])
    # Generate 2D histogram of linkage coordinates
    Nz, nx, ny = np.histogram2d(Z[:,0], Z[:,1], bins = R, density = False)
    # Normalize histogram and tag entries with eps
    Nz = Nz/sample_num + eps
    # Dimension reduce to R(R+1)/2 (omit zero values below diagonal of Nz)
    Nz = np.triu(Nz, k = 0)
    L = Nz[Nz != 0]
    return L

# initialize variables
L = list()

# set paths
inputpath = './HLR/input/'

# Initialize linkage hierarchical model
linkage = agglom(affinity='euclidean', compute_full_tree='auto',
                        connectivity=None, distance_threshold=None,
                        linkage='complete', memory=None, n_clusters=2,
                        pooling_func='deprecated')

#import data set
with open(inputpath + 'X.txt') as file:
        XX = np.array([[float(digit) for digit in line.split()] for line in file])
# import sample ID labels
ix = np.genfromtxt(inputpath + 'ix.txt')

# figure out number of samples in the data set
lastindx = len(ix)-2
Ns = len(np.unique(ix[0:lastindx]))
idx = 0

# extract Linkages from input
for j in range(Ns):
    idx=np.where(ix[0:lastindx] == j+1)
    X = XX[idx,:]
    X = X[0,:,:]
    L.append(get_linkages(X, linkage))
L = np.asarray(L)

# write to file
np.savetxt(inputpath + 'linkages.txt', L, fmt='%.8f',delimiter='\t',newline='\n')
