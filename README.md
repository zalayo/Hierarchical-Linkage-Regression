# Hierarchical-Linkage-Regression

A blind method for estimating number of clusters in a multidimensional dataset. 

## Background

This is not a clustering algorithm, but enables estimation of the cluster number of a dataset if not known *a priori*. The explicit value is often needed as an input parameter to other automated clustering methods, and can be useful for data discovery.

The method operates on the hypothesis that natural organization of data into clusters is reflected intrinsically in the hierarchical relationship of data points to one another upon partitioning of the dataset, and is therefore not dependent upon the specific values of the data, nor their absolute separations, but only on their relative ranking within the partitioned set [1]. 

The method generates linkage hierarchies from the dataset and creates new feature vectors from those hierarchies. The feature vectors are then fed to a regression model (e.g. neural network, SVM, etc.). Training of the regression model can be performed using synthetic randomly clustered data, or other forms of clustered data for which cluster count (ground truth) is known.

This repository provides the basic code to perform hierarchical linkage regression, including training and testing a model. Sample datasets are also provided.

## Software and File Organization

The code is written for Python 3, and has external dependencies on the **numpy, matplotlib, sklearn, pickle** and **joblib** libraries, as well as dependency on the internal **libHLR** module. The project file structure necessitates creation of an **./$HLR** project directory, under which the program files are kept. The subdirectories of ./$HLR must include **./$HLR/input/**, **./$HLR/output/** and **./$HLR/model/**. The ./$HLR/input/ directory serves as a working directory where executable programs will look for input files. It is recommended the user create separate storage folders to store data and files they do not want to be overwritten.

## Executable files

1. **HLRdata.py**: Generate randomly clustered dataset using random number generator. Clustering parameters and their descriptions are found in the header of the file. The parameters can be adjusted by the user to create a customized dataset.
- *Inputs*:
    - **None**
- *Outputs*:
    - **X.txt** - a (m-sample)x(n-dimensional) dataset that contains data with 1 or more clustering instances. For training purposes X must contain multiple instances of clustered data e.g. X = [X1 X2 ... XK] where for the k-th instance, Xk is itself a (m[k] x n) dimensional matrix containing one or more clusters. Saved to ./$HLR/input/
    - **kx.txt** - a label vector of length (m x n) corresponding to each of the K clustering instances in X. For the k-th instance, the label has the value k applied to the m[k] samples in that instance. Saved to ./$HLR/input/
    - **cx.txt** - a label vector of length (m x n) with unique label applied to each cluster in the k-th instance. The labels roll over for each new instance. Saved to ./$HLR/input/
    - **y.txt** - a vector of length K, corresponding to cluster number ground truth labels for the K clustering instances in X. Saved to ./$HLR/input/
    
2. **HLRlinkages.py**:  Generate linkage hierarchies from input data set and extract feature matrix for regression model
- *Inputs*: 
    - **X.txt** -  Loaded from ./$HLR/input/
    - **kx.txt** - Loaded from ./$HLR/input/  (Note: user must provide the kx file of instance labels if using their own source of data for X, otherwise the software will assume X represents a single clustering instance)
- *Outputs*: 
    - **linkages.txt** - matrix containing hierarchical linkage features for regression. Saved to ./$HLR/input/ 
- *Parameters*: 
    - **R** - parameter that sets number of bins used by the 2D histogram to generate feature matrix (default: R = 40)
    - **distance** and **linktype** (equivalent to 'affinity' and 'linkage' specified in [agglomerative clustering](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html#sklearn.cluster.AgglomerativeClustering))
    
3. **HLRtrain.py**: Train a regression model based on linkage hierarchies
- *Inputs*: 
    - **linkages.txt** - Loaded from ./$HLR/input/
    - **y.txt** - Loaded from ./$HLR/input/
- *Outputs*: 
    - **regression_model.sav** - trained regression model. Saved to "./$HLR/model/" 
    - **output_train.txt** - training model cluster number estimate paired with ground truth labels, sorted in ascending order. Saved to ./$HLR/output/ 
- *Parameters*: 
    - dependent on regression model used - see [scikit-learn documentation](https://scikit-learn.org/stable/documentation.html) for more information. [Feedforward neural network](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html#sklearn.neural_network.MLPRegressor) and [support vector machine](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html#sklearn.svm.SVR) implementations are provided, although the user is free to experiment with other regression models

4. **HLRtest.py**: Test a hierarchical linkage regression model
- *Inputs*: 
    - **linkages.txt** - Loaded from ./$HLR/input/
    - **y.txt** - Loaded from ./$HLR/input/
- *Outputs*: 
    - **output_test.txt** - test cluster number estimate paired with ground truth labels, sorted in ascending order. Saved to ./$HLR/output/ 

5. **HLRestimate.py**: Estimate number of clusters in data set with unknown cluster number
- *Inputs*: 
    - **linkages.txt** - loaded from ./$HLR/input/
- *Outputs*: 
    - **output.txt** - cluster number estimate. Saved to ./$HLR/output/

6. **HLRplot.py**: Visualization tool for plotting clustered data projected onto 3-dimensions.

- **This program requires execution from the COMMAND LINE or TERMINAL** using Python 3 
    - e.g. '>> python $(path to HLRplot.py)/HLRplot.py', or 
    
    '       >> python HLRplot.py' if executing from the program directory. 
    
    The input data and labels will be loaded from ./$HLR/input/. 

- The interactive program will prompt the user to first specify whether they want principal component analysis performed. Selecting the PCA option will plot the first 3 principal components (i.e. those with the largest covariance matrix eigenvalues), which sometimes allows for better visualization of individual clusters than the original dataset. The next prompt will ask which clustering instance to plot (ranging from 1 to K). Once the plot is closed, the user then has a choice to continue plotting other instances or exit the program.  

## References
[1] Zalay, O. *Blind method for inferring cluster number in multidimensional data sets by regression on linkage hierarchies generated from random data.* Submitted to PLOS One, Aug 2019
