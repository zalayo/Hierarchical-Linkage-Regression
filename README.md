# Hierarchical-Linkage-Regression

A blind method for estimating number of clusters in a multidimensional dataset. This is not a clustering algorithm, but enables inference of cluster number, which is often needed as an input to other automated clustering methods, and can be useful for data discovery.

The method operates on the hypothesis that natural organization of data into clusters is reflected intrinsically in the hierarchical relationship of data points to one another upon partitioning of the dataset, and is therefore not dependent upon the specific values of the data, nor their absolute separations, but only on their relative ranking within the partitioned set. 

The method generates linkage hierarchies from the dataset and creates new feature vectors from those hierarchies. The feature vectors are then fed to a regression model (e.g. neural network, SVM, etc.). Training of the regression model can be performed using synthetic randomly clustered data, or other forms of clustered data.

This repository provides the basic code to perform hierarchical linkage regression, including training and testing a model. Sample datasets are also provided.
