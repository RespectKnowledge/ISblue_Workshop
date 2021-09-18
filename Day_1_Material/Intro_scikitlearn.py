# -*- coding: utf-8 -*-
"""
Created on Sat Sep 18 13:03:57 2021

@author: Abdul Qayyum
"""

###### Introduction of scikit-learn for machine learning classifier
# main website https://scikit-learn.org/stable/
# installation pip install -U scikit-learn


# Introduction of scikit-learn
# The purpose of this guide is to illustrate some of the main features 
# that scikit-learn provides. It assumes a very basic working knowledge 
# of machine learning practices (model fitting, predicting, cross-validation, etc.). 


# Scikit-learn is an open source machine learning library that supports 
# supervised and unsupervised learning. 
# It also provides various tools:
# 1. model fitting, 
# 2. data preprocessing, 
# 3. model selection 
# 4. evaluation, 
# 5. many other utilities.

# Quick start of scikit-learn
# Scikit-learn provides dozens of 
# built-in machine learning algorithms and models, called estimators.
# Each estimator can be fitted to some data using its fit method.

# Here is a simple example where we fit a RandomForestClassifier 
# to some very basic data:
    
    
#import random forest classifier
from sklearn.ensemble import RandomForestClassifier
classifier=RandomForestClassifier(random_state=0)
# define arbitrary dataset
X=[[1,2,3],[22,34,45]] #2x3(number of samplesxnumber of features)
#define label
y=[1,0] # two samples means we have 2 labels, classes of each sample, sample1=1,sample2=class2
# fit function is an object and used for training the classifiers
classifier.fit(X,y) # parameters of fit functions are feature matrix(X) and label y

#### Gneral explanation of estimator or classifier in sklearn
# The fit method generally accepts 2 inputs:

# 1.The samples matrix (or design matrix) X. 
# The size of X is typically (n_samples, n_features), 
# which means that samples are represented as rows and features are represented as columns.

# 2.The target values y which are real numbers for regression tasks, 
# or integers for classification (or any other discrete set of values). 

# 3.For unsupervized learning tasks, y does not need to be specified. 
# y is usually 1d array where the i th entry corresponds to the target 
# of the i th sample (row) of X.

# 4.Both X and y are usually expected to be numpy arrays 
# or equivalent array-like data types, 
# though some estimators work with other formats such as sparse matrices. 

## testining or validation the model
# Once the estimator is fitted, 
# it can be used for predicting target values of new data. 
# You don’t need to re-train the estimator:
classifier.predict(X)  # predict classes of the training data
#### testining or predicting on the new dataset
classifier.predict([[4, 5, 6], [14, 15, 16]])  # predict classes of new data


##################### Transformation of Dataset in sklearn ##############

#https://scikit-learn.org/stable/modules/preprocessing.html#standardization-or-mean-removal-and-variance-scaling

# 1. Standardization, or mean removal and variance scaling
# Standardization of datasets is a common requirement 
# for many machine learning estimators implemented in scikit-learn;
# they might behave badly if the individual features do not 
# more or less look like standard normally distributed data: 
# Gaussian with zero mean and unit variance.

from sklearn import preprocessing
import numpy as np

X_train=np.array([[1.,-1.,2.],
                  [2.,0.,1.],
                  [0.,1.,-1.]])

scaler=preprocessing.StandardScaler().fit(X_train)
scaler

scaler.mean_
scaler.scale_

X_scaled=scaler.transform(X_train)
X_scaled 
###zero mean and unit stdnard deviation
X_scaled.mean(axis=0)
X_scaled.std(axis=0)
###### another example
# dataset Transformers and pre-processors
from sklearn.preprocessing import StandardScaler
X = [[0, 15],[1, 10]]
# scale data according to computed scaling values
StandardScaler().fit(X).transform(X)

#2. Scaling features to a range
# An alternative standardization is scaling features to lie between 
# a given minimum and maximum value, 
# often between zero and one, 
# or so that the maximum absolute value of each feature is scaled to unit size. 
# This can be achieved using MinMaxScaler or MaxAbsScaler, respectively.

min_max_scaler=preprocessing.MinMaxScaler()
x_train_min_max=min_max_scaler.fit_transform(X_train)

print(x_train_min_max)
print(x_train_min_max.mean(axis=0))
print(x_train_min_max.mean())
print(x_train_min_max.std(axis=0))
print(x_train_min_max.std())
## similary min max can be applied on unseen test dataset
X_test = np.array([[-3., -1.,  4.]])
X_test_minmax = min_max_scaler.transform(X_test)
print(X_test_minmax)

# MaxAbsScaler works in a very similar fashion, 
# but scales in a way that the training data lies within the range [-1, 1] 
# by dividing through the largest maximum value in each feature. 
# It is meant for data that is already centered at zero or sparse data.

max_abs_scaler = preprocessing.MaxAbsScaler()
X_train_maxabs = max_abs_scaler.fit_transform(X_train)
X_train_maxabs

X_test = np.array([[ -3., -1.,  4.]])
X_test_maxabs = max_abs_scaler.transform(X_test)
X_test_maxabs
max_abs_scaler.scale_

#3. Normalization
# Normalization is the process of scaling individual samples to have unit norm. 
# This process can be useful if you plan to use a quadratic form such as 
# the dot-product or any other kernel to quantify the similarity of any pair 
# of samples.

# This assumption is the base of the Vector Space Model often used 
# in text classification and clustering contexts.

# The function normalize provides a quick and easy way to perform 
# this operation on a single array-like dataset, either using the l1, l2, 
# or max norms:
X = [[ 1., -1.,  2.],[ 2.,  0.,  0.],[ 0.,  1., -1.]]

X_normalized = preprocessing.normalize(X, norm='l2')

print(X_normalized)

print(X_normalized.min())
print(X_normalized.max())

X_normalized1 = preprocessing.normalize(X)

print(X_normalized1)

print(X_normalized1.min())
print(X_normalized1.max())


X_normalizedmax = preprocessing.normalize(X,'max')

print(X_normalizedmax)

print(X_normalizedmax.min())
print(X_normalizedmax.max())
####### another way to define normlization of training and testing
X_normalizedmaxf = preprocessing.Normalizer().fit(X)
X_normalizedmaxf.transform(X)  # training
X_normalizedmaxf.transform([[-1.,  1., 0.]]) #testing

############## standard pipeline for classification in scikit learn #######################
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# define pipline object with data normalization and model
pipe=make_pipeline(StandardScaler(),
                   LogisticRegression()
                   )

# load the iris dataset and split it into train and test sets
X, y = load_iris(return_X_y=True)
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=0)
### fit function for training
pipe.fit(X_train,y_train)
# we can now use it like any other estimator for prediction and get accuracy
accuracy_score(pipe.predict(X_test), y_test)






