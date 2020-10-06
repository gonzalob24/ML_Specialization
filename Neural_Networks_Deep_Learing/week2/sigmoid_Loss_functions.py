#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: gonzalobetancourt
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def sigmoid(x):
    """
    Compute the sigmoid of x

    Arguments:
    x -- A scalar or numpy array of any size

    Return:
    s -- sigmoid(x)

    Parameters
    ----------
    x : Scalar or numpy array 
        it can be any size.

    Returns
    -------
    s : float
        sigmoid output.

    """
    
    s = 1 / (1 + np.exp(-x))
    
    return s

## Sigmoid Gradient
def sigmoid_derivative(x):
    """
     Compute the gradient (also called the slope or derivative) of the 
     sigmoid function with respect to its input x.
     You can store the output of the sigmoid function into variables and 
     then use it to calculate the gradient.


    Parameters
    ----------
    x : A scalar or numpy array

    Returns
    -------
    ds : computed gradient
        DESCRIPTION.

    """
    s = sigmoid(x)
    ds = s * (1 - s)
    
    return ds


# Reshaping an image or unrilling

def image2vector(image):
    """
    
    Parameters
    ----------
    image : numpy array of shape (length, height, depth)

    Returns
    -------
    v : vector of shape (length*height*depth, 1)

    """
    
    v = image.reshape(image.shape[0], image.shape[1], image.shape[2])
    
    return v


# Normalizing or fearure scalling

def normalizeRows(x):
    """
    Implement a function that normalizes each row of the matrix x (to have unit length).

    Parameters
    ----------
    x : numpy matrix of shape (n, m)

    Returns
    -------
    x : normalized (by row) numpy matrix. You are allowed to modify x

    """
    
    x_norm = np.linalg.norm(x, axis=1, keepdims=True)
    
    x = x/x_norm

    return x


# softmax function
# A softmax is kind of like a normalizing function used when your algorithm 
# needs to classify two or more classes. 

def softmax(x):
    """
    Calculates the softmax for each row of the input x.

    Parameters
    ----------
    x : numpy matrix of shape (m,n)

    Returns
    -------
    s : A numpy matrix equal to the softmax of x, of shape (m,n)

    """
    
    # Apply exp() element-wise to x.
    x_exp = np.exp(x)

    # Create a vector x_sum that sums each row of x_exp.
    x_sum = np.sum(x_exp, axis=1, keepdims=True)
    
    # Compute softmax(x) by dividing x_exp by x_sum, automatically use numpy broadcasting.
    s = x_exp/x_sum
    
    return s

# L1 and L2 loss function
# The loss is used to evaluate the performance of your model. 
# The bigger your loss is, the more different your predictions 
#( ŷ ) are from the true values ( y). In deep learning, you use optimization 
#algorithms like Gradient Descent to train your model and to minimize the cost.
# L1 loss is defined as
# sum(|yhat - y|)

def L1(yhat, y):
    """
    Parameters
    ----------
    yhat : vector of size m (predicted labels)
    
    y : vector of size m (true labels)

    Returns
    -------
    loss : value of the L1 loss function 

    """
    
    loss = np.sum(np.abs(yhat - y))
    
    return loss


# sum(|yhat - y|)^2
def L2(yhat, y):
    """

    Parameters
    ----------
    yhat : vector of size m (predicted labels)

    y : vector of size m (true labels)

    Returns
    -------
    loss : the value of the L2 loss

    """
    
    ### START CODE HERE ### (≈ 1 line of code)
    loss = np.sum(np.dot(np.abs(yhat-y), np.abs(yhat-y)))
    ### END CODE HERE ###
    
    return loss