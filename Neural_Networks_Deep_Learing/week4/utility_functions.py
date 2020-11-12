#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Project Name: Deep Learning Model for Image Classification
    
Project Description: Building a Deep Learning Model From Scratch
    
This file will have all of the modules I will use to implement a Deep NN model

@author: gonzalobetancourt

"""

import numpy as np 
import matplotlib.pyplot as plt
import h5py

def sigmoid(Z):
    """
    Implements the Sigmoid function

    Parameters
    ----------
    Z : numpy array.
        Can be any shape.

    Returns
    -------
    A : numpy array.
        Same shape as sigmoid(Z).
    cache : numpy array -- Z
        cache Z used during backpropagation.
    """
    
    A = 1 / (1 + np.exp(-Z))
    cache = Z
    
    return A, cache

def relu(Z):
    """
    Implements the RELU function

    Parameters
    ----------
    Z : numpy array
        Can be any shape.

    Returns
    -------
    A : numpy array.
        Same shape as Z. A -- post activation parameter.
    cache : dictionary
        Contains A -- used for calculating back propagation.
    """
    A = np.maximum(0, Z)
    
    assert(A.shape == Z.shape)
    
    cache = Z
    return A, cache


def relu_backward(dA, cache):
    """
    Implements the backward propagation for a single RELU unit.

    Parameters
    ----------
    dA : numpy array
        post-activation gradient, of any shape.
    cache : numpy array
        'Z' stored for computing backward propagation.

    Returns
    -------
    dZ : numpy array.
        Fradient of the cost with respect to Z
    """
    
    Z = cache
    # Converting dZ to a correct object.
    dZ = np.array(dA, copy=True)
    
    # When z <= 0, you should set dZ to 0 as well. 
    dZ[Z <= 0] = 0
    
    assert (dZ.shape == Z.shape)
    
    return dZ

def sigmoid_backward(dA, cache):
    """
    Implements the backward propagation for a single SIGMOID unit.

    Parameters
    ----------
    dA : numpy array
        post-activation gradient, of any shape.
    cache : numpy array
        'Z' stored for computing backward propagation.

    Returns
    -------
    dZ : numpy array.
        Gradient of the cost with respect to Z
    """
    
    Z = cache
    
    s = 1/(1+np.exp(-Z))
    dZ = dA * s * (1-s)
    
    assert (dZ.shape == Z.shape)
    
    return dZ


def load_data():
    train_dataset = h5py.File('datasets/train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels

    test_dataset = h5py.File('datasets/test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels

    classes = np.array(test_dataset["list_classes"][:]) # the list of classes
    
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes


def initialize_parameters(n_x, n_h, n_y):
    """
    Initialize the parameters

    Parameters
    ----------
    n_x : int
        size of the input layer.
    n_h : int
        size of the hidden layer.
    n_y : int
        size of the output layer.

    Returns
    -------
    parameters : dictionary
        W1 -- weight matrix of shape (n_h, n_x)
        b1 -- bias vector of shape (n_h, 1)
        W2 -- weight matrix of shape (n_y, n_h)
        b2 -- bias vector of shape (n_y, 1).

    """
    # Set a random seed to get same results
    np.random.seed(1)
    
    W1 = np.random.randn(n_h, n_x)*0.01
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h)*0.01
    b2 = np.zeros((n_y, 1))
    
    assert(W1.shape == (n_h, n_x))
    assert(b1.shape == (n_h, 1))
    assert(W2.shape == (n_y, n_h))
    assert(b2.shape == (n_y, 1))
    
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    
    return parameters     


def initialize_parameters_deep(layer_dims):
    """
    Initialize parameters for a deep NN

    Parameters
    ----------
    layer_dims : list
        containins the dimensions of each layer in the network.

    Returns
    -------
    parameters : dictionary
        Wl -- weight matrix of shape (layer_dims[l], layer_dims[l-1])
        bl -- bias vector of shape (layer_dims[l], 1).

    """
    
    np.random.seed(1)
    parameters = {}
    
    # number of layers in the network
    L = len(layer_dims)

    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) / np.sqrt(layer_dims[l-1])
        # parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * 0.01
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
        
        assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l-1]))
        assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))

    return parameters


def linear_forward(A, W, b):
    """
    Implements the linear part of a layer's forward propagation.

    Parameters
    ----------
    A : numpy array.
        Activations from pervious layer or input data.
        (size of previous layer, number of examples).
    W : numpy array
        weights matrix (size of current layer, size of previous layer).
    b : numpy vector.
        (size of the current layer, 1).

    Returns
    -------
    Z : Numpy array
        the input of the activation function, also called pre-activation parameter.
    cache : dictionary
        containing "A", "W" and "b" ; stored for computing the backward pass.

    """
    
    # Z = W.dot(A) + b
    Z = np.dot(W, A) + b
    
    assert(Z.shape == (W.shape[0], A.shape[1]))
    cache = (A, W, b)
    
    return Z, cache


def linear_activation_forward(A_prev, W, b, activation):
    """
    Implements the forward propagation of the linear activation layer

    Parameters
    ----------
    A_prev : numpy array
        activations from previous layer or input data: (size of previous layer, number of examples).
    W : numpy array
        weights matrix -- numpy array of shape (size of current layer, size of previous layer).
    b : numpy array
        bias vector -- numpy array of shape (size of the current layer, 1).
    activation : string
        activation to be used in this layer -- "sigmoid" or "relu".

    Returns
    -------
    A : numpy array.
        output of the activation function, also called the post-activation value 
    cache : tuple
        contains "linear_cache" and "activation_cache" -- stored for computing the backward pass

    """
    
    #Z, linear_cache = linear_forward(A_prev, W, b)
    # linear_cache --> stores A_prev, w, b
    # activation_cache --> stores Z
    if activation == "sigmoid":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)
    elif activation == "relu":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)

    assert(A.shape == (W.shape[0], A_prev.shape[1]))
    
    cache = (linear_cache, activation_cache)
    
    return A, cache


def deep_model_forward(X, parameters):
    """
    Implements the forward propagation [linear relu]*(L-1) -- linear sigmoid 

    Parameters
    ----------
    X : numpy array
        data shape (input size, number of examples).
    parameters : dictionary
        output of initialize_parameters_deep().

    Returns
    -------
    AL : nunmpy array.
        last post-activation value.
    cache : list
        contains every cache of linear_activation_forward() (there are L-1 of them, indexed from 0 to L-1)
    """
    
    caches = []
    # set A0 = X
    A = X
    
    # number of layers in model
    L = len(parameters) // 2
    
    # [Linear Relu] * (L - 1). Add cache to caches list
    for layer in range(1, L):
        A_prev = A
        A, cache = linear_activation_forward(A_prev, parameters["W" + str(layer)], parameters["b" + str(layer)], "relu")
        caches.append(cache)
    
    # linear sigmoid. Add cache to caches list
    AL, cache = linear_activation_forward(A, parameters["W" + str(L)], parameters["b" + str(L)], "sigmoid")
    caches.append(cache)
    
    assert(AL.shape == (1, X.shape[1]))
    
    return AL, caches


def compute_cost(AL, Y):
    """
    Implements the cost function defined by 
    −1/𝑚∑=(𝑦(𝑖)*log(𝑎[𝐿](𝑖)) + (1−𝑦(𝑖))*log(1−𝑎[𝐿](𝑖))).

    Parameters
    ----------
    AL : numpy array
        probability vector corresponding to label predictions, shape (1, number of examples).
    Y : numpy array
        true "labels" vector (for example: containing 0 if false, 1 if true), shape (1, number of examples).

    Returns
    -------
    cost : float.
        cross-entropy cost
    """
    
    # set the number of labels
    m = Y.shape[1]
    
    # compute the loss from AL to Y
    # cost = -(1/m) * np.sum(np.multiply(Y, np.log(AL)) + np.multiply(1 - Y, np.log(1 - AL)))
    cost = (1/m) * (-np.dot(Y, np.log(AL).T) - np.dot(1 - Y, np.log(1 - AL).T))

    # cost = -(1/m)*(np.dot(Y, np.log(AL).T) + np.dot(1-Y, np.log(1-AL).T))
    
    # makes sure cost's shape is what we expect (e.g. this turns [[17]] into 17).
    cost = np.squeeze(cost)
    
    assert(cost.shape == ())
    
    return cost


def linear_backward(dZ, cache):
    """
    Implements the lienar portion of backward propagation for a single layer

    Parameters
    ----------
    dZ : numpy array
        Gradient of the cost with respect to the linear output (of current layer l).
    cache : tuple
        tuple of values (A_prev, W, b) coming from the forward propagation in the current layer.

    Returns
    -------
    dA_prev : numpy array.
        Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev.
    dW : numpy array
        Gradient of the cost with respect to W (current layer l), same shape as W
    db : numpy array
        Gradient of the cost with respect to b (current layer l), same shape as b
    """
    
    A_prev, W, b = cache
    m = A_prev.shape[1]

    
    dW = 1./m * np.dot(dZ, A_prev.T)
    db = 1./m * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)
    
    # check the dimensions
    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)
    
    return dA_prev, dW, db
    
def linear_activation_backward(dA, cache, activation):
    """
    Implements the backward propagation for the LINEAR ACTIVATION layer.

    Parameters
    ----------
    dA : numpy array.
        post-activation gradient for current layer l.
    cache : tupl.
        values (linear_cache, activation_cache) we store for computing backward propagation efficiently
    activation : string.
        the activation to be used in this layer, "sigmoid" or "relu"

    Returns
    -------
    dA_prev : numpy array.
        Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    dW : numpy array
        Gradient of the cost with respect to W (current layer l), same shape as W
    db : numpy array
        Gradient of the cost with respect to b (current layer l), same shape as b
    """
        
    linear_cache, activation_cache = cache
    
    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
        
    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
    
    return dA_prev, dW, db


def deep_model_backward(AL, Y, caches):
    """
    Implement the backward propagation for the [LINEAR->RELU] * (L-1) -> LINEAR -> SIGMOID group

    Parameters
    ----------
    AL : numpy array.
        probability vector, output of the forward propagation (L_model_forward())
    Y : numpy array.
        true "label" vector (containing 0 if False, 1 if True)
    caches : list.
        contains every cache of linear_activation_forward() with "relu" (it's caches[l], for l in range(L-1) i.e l = 0...L-2)
        the cache of linear_activation_forward() with "sigmoid" (it's caches[L-1])

    Returns
    -------
    grads : dictionary.
        grads["dA" + str(l)] = ... 
        grads["dW" + str(l)] = ...
        grads["db" + str(l)] = ...

    """
    grads = {}
    # the number of layers
    L = len(caches) 
    m = AL.shape[1]
    # after this line, Y is the same shape as AL
    Y = Y.reshape(AL.shape) 
    
    # Initializing the backpropagation
    dAL = -(np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    
    # Lth layer (sigmoid - linear) gradients
    current_cache = caches[-1]
    # grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, current_cache, 'sigmoid')
    grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = linear_backward(sigmoid_backward(dAL, current_cache[1]), current_cache[0])

    # Loop from l=L-2 to l=0
    for l in reversed(range(L-1)):
        # lth layer: (rely - linear) gradients.
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_backward(relu_backward(grads["dA" + str(1 + l)], current_cache[1]), current_cache[0])

        #dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l + 1)], current_cache, 'relu')
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp

    return grads


def update_parameters(parameters, grads, learning_rate):
    """
    Update parameters using gradient descent

    Parameters
    ----------
    parameters : dictionary.
        containis your parameters 
    grads : dictionary.
        contains your gradients, output of L_model_backward
    learning_rate : float
        set the learning rate

    Returns
    -------
    parameters : dictionary.
        parameters["W" + str(l)] = ... 
        parameters["b" + str(l)] = ...
    """
    
    # number of layers in the neural network
    L = len(parameters) // 2 

    # Update rule for each parameter
    for layer in range(L):
        parameters["W" + str(layer + 1)] = parameters["W" + str(layer + 1)] - learning_rate * grads["dW" + str(layer + 1)]
        parameters["b" + str(layer + 1)] = parameters["b" + str(layer + 1)] - learning_rate * grads["db" + str(layer + 1)]
    
    return parameters


def predict(X, y, parameters):
    """
    This function is used to predict the results of a  L-layer neural network.
    
    Parameters:
    X : numpy array.
        data set of examples
    y : numpy array.
        true labels
    parameters : dictionary.
        parameters of the trained model
    
    Returns:
    p : numpy array.
        predictions for the given dataset X
    """
    
    m = X.shape[1]
    n = len(parameters) // 2 # number of layers in the neural network
    p = np.zeros((1,m))
    
    # Forward propagation
    probabilities, caches = deep_model_forward(X, parameters)

    
    # convert probas to 0/1 predictions
    for i in range(0, probabilities.shape[1]):
        if probabilities[0,i] > 0.5:
            p[0,i] = 1
        else:
            p[0,i] = 0
    
    #print results
    #print ("predictions: " + str(p))
    #print ("true labels: " + str(y))
    print("Accuracy: "  + str(np.sum((p == y)/m)))
        
    return p























