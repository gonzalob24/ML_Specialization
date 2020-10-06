#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: gonzalobetancourt
"""

import numpy as np
from planar_utils import sigmoid

def set_leyer_sizes(X, Y):
    """
    Set the sizes of the NN layers

    Parameters
    ----------
    X : dataset features 
        input size and number of examples.
    Y : dataset labels
        output size and number of examples.

    Returns
    -------
    n_x : size of input layer.
    n_h : size of hidden layer.
    n_y : size of output layer.
    """
    
    n_x = X.shape[0]
    # This size is hardcoded
    n_h = 4
    n_y = Y.shape[0]
    
    return (n_x, n_h, n_y)


def initialize_parameters(n_x, n_h, n_y):
    """
    

    Parameters
    ----------
    n_x : int
        size of input layer.
    n_h : int
        size of hidden layer.
    n_y : int
        size of output layer.

    Returns
    -------
    parameters : model parameters.

    """
    
    # For testing set a random seet
    np.random.seed(2)
    
    # randomly initialize paramaters
    W1 = np.random.randn(n_h, n_x) * 0.01 
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros((n_y, 1))
    
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    
    return parameters

def forward_propagation(X, parameters):
    """
    Perform forward propagation 

    Parameters
    ----------
    X : numpy array
        input data size (n_x, m_size).
    paramaters : dict
        dictionary containing parameters.

    Returns
    -------
    A2 : sigmoid output of the second activation function.
    cache : dictionary containing Z1, A1, Z2, A2.

    """
    
    # Get randomly initialzied parameters 
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    
    # Use equations to implement forward propagation
    Z1 = np.dot(W1, X) + b1
    A1 = np.tanh(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)
    
    # cache the equations and return cache and A2
    
    cache = {"Z1": Z1,
             "A1": A1,
             "Z2": Z2,
             "A2": A2}
    
    return A2, cache



def calc_cost(A2, Y, parameters):
    """
    Compute the cost for backward propagation.

    Parameters
    ----------
    A2 : float
        size (1, m_samples.
    Y : numpy array
        true labels size (1, m_samples.
    parameters : dict
        dictionary containg parameters W1, b1, W2, b2.

    Returns
    -------
    cost : cross-entropy, using the equation.

    """
    # m number of samples
    m = Y.shape[1]
    
    # using the equation in class
    logprobs = np.multiply(np.log(A2), Y) + np.multiply((1 - Y), np.log(1 - A2))
    cost = -np.sum(logprobs) / m
    
    # put cost in the correct dimensions
    cost = float(np.squeeze(cost))
    
    return cost


def backward_propagation(parameters, cache, X, Y):
    """
    Implement backward propagation.

    Parameters
    ----------
    parameters : dict
        dictionary containg parameters W1, b1, W2, b2.
    cache : dict
        dictionary containing Z1, A1, Z2, A2.
    X : numpy array
        input data shape (2, m_samples).
    Y : numpy array
        true labels size (1, m_samples).

    Returns
    -------
    gradient : dictionary containing gradients with respect to different parameters.

    """
    
    # Samples
    m = X.shape[1]
    
    # Get W1 and W2 from parameters
    W1 = parameters["W1"]
    W2 = parameters["W2"]
    
    # Get A1 and A2 from cache
    A1 = cache["A1"]
    A2 = cache["A2"]
    
    
    # Backward Prop
    # calc dW1, db1, dW2, db2 
    
    # USe the equarions covered in class
    dZ2 = A2 - Y
    dW2 = (1/m) * np.dot(dZ2, A1.T)
    db2 = (1/m) * np.sum(dZ2, axis=1, keepdims=True)
    # dZ1 = np.dot(W2.T, dZ2) * (1 - np.power(A1, 2))
    dZ1 = np.multiply(np.dot(W2.T, dZ2), (1 - np.power(A1, 2)))
    dW1 = (1/m) * np.dot(dZ1, X.T)
    db1 = (1/m) * np.sum(dZ1, axis=1, keepdims=True)

    # Object gradients 
    gradients = {"dW1": dW1,
                 "db1": db1,
                 "dW2": dW2,
                 "db2": db2}
    
    return gradients


def update_parameters(parameters, gradients, learning_rate = 1.2):
    """
    update the parametes to calcualte gradient descent.

    Parameters
    ----------
    parameters : dict
        dictionary containg parameters W1, b1, W2, b2.
    gradients : dict
        dictionary containing gradients with respect to different parameters.
    learning_rate : flaot, optional
        learing-rate, the default is 1.2.

    Returns
    -------
    parameters : dictionary containg parameters W1, b1, W2, b2.

    """
    
    
    # Get parameters
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    
    # Get the gradients
    dW1 = gradients["dW1"]
    db1 = gradients["db1"]
    dW2 = gradients["dW2"]
    db2 = gradients["db2"]
    
    # Update parameters
    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1
    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2
    
    # return updated parameters
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    
    return parameters


def predict(parameters, X):
    """
    Make predicitons

    Parameters
    ----------
    paramaters : dict
        model parameters.
    X : numpy array
        input data size (n_x, m_samples).

    Returns
    -------
    predictions : vector of predictions of model (red: 0 / blue: 1).

    """
    
    # Compute probabilities using forwar propagation using 0.5 as threshold 
    # 1 if > 0.5 else 0
    A2, cache = forward_propagation(X, parameters)
    
    # Round predictions to whole int
    predictions = np.round(A2)
    
    return predictions


