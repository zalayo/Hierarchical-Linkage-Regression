import os
import numpy as np
from sklearn.cluster import AgglomerativeClustering as agglom


class randclust:
    ## Generate Random Clustered Data for training and testing
    ############################################################################
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
    minsamples = 20
    ## maximum number of samples allowed per cluster
    maxsamples = 500
    ############################################################################

    # instantiate object with total number of clustering intances
    def __init__(self, value):
        np.random.seed()
        self.K = value

    # generate ith cluster in kth instance
    def cluster(self, clustsz, i):
        xkc = np.random.uniform(self.lb, self.ub, size = (1,self.dim))
        sdk = np.random.uniform(self.sdl, self.sdu, size = (1,self.dim))
        xki = np.multiply(np.random.randn(clustsz[0], self.dim), sdk)
        Xki = xkc + xki
        indx_ki = (i+1)*np.ones((clustsz[0],))
        return Xki, indx_ki

    # generate kth clustering instance
    def instance(self, k):
        clustnum_k = np.random.randint(1, self.maxclust, size=1)
        Xk = np.asarray([])
        indx_i = np.asarray([])
        for i in range(clustnum_k[0]):
            clustsz = np.random.randint(self.minsamples,
                                        self.maxsamples, size=1)
            Xki, indx_ki = self.cluster(clustsz, i)
            if i != 0:
                Xk = np.vstack((Xk, Xki))
            if i == 0:
                Xk = Xki
            indx_i = np.concatenate((indx_i, indx_ki))
        indx_k = (k+1)*np.ones((np.shape(Xk)[0],))
        return Xk, indx_k, indx_i, clustnum_k

    # generate dataset of K clustering instances
    def dataset(self):
        X, y, kx, cx = np.asarray([]), np.asarray([]), \
                        np.asarray([]), np.asarray([])
        for k in range(self.K):
            Xk, indx_k, indx_i, clustnum_k = self.instance(k)
            if k != 0:
                X = np.vstack((X, Xk))
            if k == 0:
                X = Xk
            y = np.concatenate((y, clustnum_k))
            kx = np.concatenate((kx, indx_k))
            cx = np.concatenate((cx, indx_i))
        y = np.asarray(y)
        return X, y, kx, cx


class linkages:
    ## Generate Hierarchical Linkage Features for Regression
    ############################################################################
    # Bin count for 2D histogram
    R = 40
    # Distance metric (e.g. 'manhattan' or 'euclidean')
    distance = 'manhattan'
    # Link type (e.g. 'complete' or 'ward')
    linktype = 'complete'
    # Definition of epsilon
    eps = np.finfo(np.float).eps
    ###########################################################################

    def __init__(self, distance, linktype):
    #   instantiate linkage model
        self.linkage = agglom(affinity = self.distance,
                            linkage = self.linktype,
                            compute_full_tree = 'auto',
                            connectivity = None,
                            distance_threshold = None,
                            memory = None,
                            n_clusters = 1)

    def get(self,X):
        # Generate linkage hierarchy and extract feature vector
        # Number of samples in X
        sample_num = np.shape(X)[0]
        # Generate linkage hierarchy
        self.linkage.fit(X)
        # Get internal node coordinates of linkage tree
        Z = np.asarray(self.linkage.children_ + 1)
        Z.reshape((self.linkage.n_leaves_ - 1, 2))
        # Generate 2D histogram of linkage coordinates
        Nz, nx, ny = np.histogram2d(Z[:,0], Z[:,1], bins = self.R,
                                                    density = False)
        # Normalize histogram and tag entries with eps
        Nz = Nz/sample_num + self.eps
        # Dimension reduce to R(R+1)/2 (omit zero values below diagonal of Nz)
        Nz = np.triu(Nz, k = 0)
        L = Nz[Nz != 0]
        return L
