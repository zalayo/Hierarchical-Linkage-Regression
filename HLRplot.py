import os
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import IncrementalPCA

## set paths
inputpath = './input/'

## Toggle PCA for better visualization of clusters
pca_flag = int(input("\nPerform PCA for cluster visualization?:" \
                            + " press 1 if YES,  and 0 if NO \n"))
# number of PCA components
N = 3

#import data set
print("\n\n")
print("\nLoading data from " + os.path.dirname(os.path.realpath(__file__)) +
                                                                inputpath[1:])
print("\nPlease be patient...this can take a while for large files...")
print("\n\n")
with open(inputpath + 'X.txt') as file:
    X = np.array([[float(digit) for digit in line.split()] for line in file])

# import clustering labels but check to see if files exist first
if not os.path.exists(inputpath + 'kx.txt'):
    kx = np.ones((np.shape(X)[0],))
else:
    kx = np.genfromtxt(inputpath + 'kx.txt')
if not os.path.exists(inputpath + 'cx.txt'):
    cx = np.ones((len(kx),))
else:
    cx = np.genfromtxt(inputpath + 'cx.txt')

# create index for plotting
indx = np.vstack((kx.astype(int), cx.astype(int)))

# get number of clustering instances
K = len(np.unique(indx[0,:]))

if pca_flag:
    # batch size for incremental PCA
    batchsz = 10
    # perform PCA for visualization of clusters
    pca = IncrementalPCA(n_components = N, batch_size = batchsz)
    X = pca.fit_transform(X)

# main loop
notdone = True
while notdone:
    instance = input("\nWhat is the clustering instance you wish to plot? ")
    instance = int(instance)
    print('\nProcessing...\n')
    # project onto 3D axes
    plt.figure(figsize=(10,8))
    ax = plt.axes(projection='3d')
    title = "Cluster plot: instance %d" % instance
    kindx = np.asarray(np.where(indx[0,:] == instance))
    cindx = np.unique(indx[1,kindx])
    #for i, target_name in zip(range(Nc), iris.target_names):
    for i in cindx:
        ax.scatter3D(X[kindx[indx[1,kindx] == i], 0],
                            X[kindx[indx[1,kindx] == i], 1],
                            X[kindx[indx[1,kindx] == i], 2],
                            label = str(i), s = 4)
        #print(np.std(X[kindx[indx[1,kindx] == i],:], axis = 0))
    plt.title(title + " of %d" % K)
    #if display_clusternum:
    #    ax.text2D(1, 1, r'y ='+str(y[instance-1]), fontsize=10, transform=ax.transAxes)
    plt.legend(loc="upper left", shadow=False, scatterpoints=1)
    plt.show()
    getuserinput = input("Want to continue?: press 1 if YES," \
                                              + " and 0 to EXIT \n\n")
    if(eval(getuserinput) == 0):
        notdone = False
        print('\nExiting...\n\n')
