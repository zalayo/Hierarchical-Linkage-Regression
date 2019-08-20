# Hierarchical-Linkage-Regression

A blind method for estimating number of clusters in a multidimensional dataset. 

## Background

This is not a clustering algorithm, but enables estimation of the cluster number if unknown. The explicit value is often needed as an input parameter to other automated clustering methods, and can be useful for data discovery.

The method operates on the hypothesis that natural organization of data into clusters is reflected intrinsically in the hierarchical relationship of data points to one another upon partitioning of the dataset, and is therefore not dependent upon the specific values of the data, nor their absolute separations, but only on their relative ranking within the partitioned set [1]. 

The method generates linkage hierarchies from the dataset and creates new feature vectors from those hierarchies. The feature vectors are then fed to a regression model (e.g. neural network, SVM, etc.). Training of the regression model can be performed using synthetic randomly clustered data, or other forms of clustered data for which cluster count (ground truth) is known.

This repository provides the basic code to perform hierarchical linkage regression, including training and testing a model. Sample datasets are also provided.

## Software and File Organization

The code is written for Python 3, and has dependencies on the numpy, matplotlib, sklearn, pickle and joblib libraries. The project file structure necessitates creation of an "./$HLR" project directory, under which the program files are kept. The subdirectories of "./$HLR" must include "./$HLR/input/", "./$HLR/output/" and "./$HLR/model/".

## Executable files
(1) HLR_linkages.py:  Generate linkage hierarchies from input data set and extract feature matrix for regression model
- Inputs: 
    - 'X.txt' - a (m-sample)x(n-dimensional) dataset that contains data with 1 or more clusters. For training purposes X must contain multiple instances of clustered data e.g. X = [X1 X2 ... XK] where for the k-th instance, Xk is itself a (mk x n) dimensional matrix. Loaded from "./$HLR/input/"
    - 'ix.txt' - an ID label vector of length (m x n) corresponding to clustering instances in X. For the k-th instance, the label has the value k applied to the mk samples in that instance. Loaded from "./$HLR/input/"
- Outputs: 
    - 'linkages.txt' - matrix containing linkage features for regression. Saved to "./$HLR/input/" 
- Parameters: 
    - 'R' - internal parameter (which can be user adjusted) that sets number of bins used by the 2D histogram to generate feature matrix
    - 'affinity' (i.e. distance metric) and 'linkage' from [agglomerative clustering](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html#sklearn.cluster.AgglomerativeClustering)

(2) HLR_train.py: Train a regression model based on linkage hierarchies
- Inputs: 
    - 'linkages.txt' - loaded from "./$HLR/input/"
    - 'cx.txt' - a vector of length K, corresponding to cluster number ground truth labels for the K clustering instances in X. Loaded from "./$HLR/input/"
- Outputs: 
    - 'regression_model.sav' - trained regression model. Saved to "./$HLR/model/" 
    - 'output_train.txt' - training model cluster number estimate paired with ground truth labels, sorted in ascending order. Saved to "./$HLR/output/" 
- Parameters: 
    - dependent on regression model used - see [scikit-learn documentation](https://scikit-learn.org/stable/documentation.html) for more information. [Feedforward neural network](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html#sklearn.neural_network.MLPRegressor) and [support vector machine](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html#sklearn.svm.SVR) implementations are provided in file

(3) HLR_test.py: Test a hierarchical linkage regression model
- Inputs: 
    - 'linkages.txt' - loaded from "./$HLR/input/"
    - 'cx.txt' - a vector of length K, corresponding to cluster number ground truth labels for the K clustering instances in X Loaded from "./$HLR/input/"
- Outputs: 
    - 'output_test.txt' - test cluster number estimate paired with ground truth labels, sorted in ascending order. Saved to "./$HLR/output/" 

(4) HLR_estimate.py: Estimate number of clusters in data set with unknown cluster number
- Inputs: 
    - 'linkages.txt' - loaded from "./$HLR/input/"
- Outputs: 
    - 'output.txt' - cluster number estimate. Saved to "./$HLR/output/" 

## References
[1] Blind method for inferring cluster number in multidimensional data sets by regression on linkage hierarchies generated from random data. Submitted to PLOS One, Aug 2019
