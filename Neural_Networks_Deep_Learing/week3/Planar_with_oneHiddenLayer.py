#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: gonzalobetancourt
"""

import numpy as np
import matplotlib.pyplot as plt
from testCases_v2 import *
import sklearn
import sklearn.datasets
import sklearn.linear_model
from planar_utils import plot_decision_boundary, sigmoid, load_planar_dataset, load_extra_datasets
from oneHiddenLayer_NN_utils import predict
from oneHL_NN_model import *

# To get consisten results
np.random.seed(1)

X, Y = load_planar_dataset()
# visualize the features
plt.scatter(X[0, :], X[1, :], c=Y, s=40, cmap=plt.cm.Spectral)

# Simple logistic regression 
logistic_classifier = sklearn.linear_model.LogisticRegressionCV()
logistic_classifier.fit(X.T, Y.T)

# Decision doundary plot for logistic classifier
plot_decision_boundary(lambda x: logistic_classifier.predict(x), X, Y)


# Let take a look at the accuracy --> under 50%
# LR does not work well when the data is not linerarly seperable
lr_pred = logistic_classifier.predict(X.T)
print('Accuracy of logistic regression: {}% percentage of correctly labelled datapoints'.format(float((np.dot(Y,lr_pred) + np.dot(1-Y,1-lr_pred))/float(Y.size)*100)))

# NN with one hidden layer
parameters = nn_model(X, Y, n_h=4, epochs=10000, print_cost=True)

# make predictions and plot the boundary
plot_decision_boundary(lambda x: predict(parameters, x.T), X, Y)
plt.title("Decision Boundary for hidden layer size " + str(4))

predictions = predict(parameters, X)
print ("Accuracy: {}%".format(float((np.dot(Y,predictions.T) + np.dot(1-Y,1-predictions.T))/float(Y.size)*100)))







