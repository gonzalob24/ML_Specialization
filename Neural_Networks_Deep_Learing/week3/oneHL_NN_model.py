#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: gonzalobetancourt
"""

from oneHiddenLayer_NN_utils import *

def nn_model(X, Y, n_h=4, epochs=10000, print_cost = False):
    """
    Implement NN with one hidden layer

    Parameters
    ----------
    X : numpy array
        shape (2, m_samples).
    Y : numpy array
        shape (1, m_samples).
    epochs : int
        number of iterations.
    print_cost : Boolean, optional
        Print the cost. The default is False.

    Returns
    -------
    parameters : parameters of the model, can be used for predictions

    """
    
    # set a random seed to get the same results
    np.random.seed(3)
    
    # set size of n_x and n_y
    n_x = set_leyer_sizes(X, Y)[0]    
    n_y = set_leyer_sizes(X, Y)[2]   
    
    # initialize parameters 
    parameters = initialize_parameters(n_x, n_h, n_y)
    
    # set up loop for forward/backward propagation and gradient descent
    
    for i in range(0, epochs):
        # forward propagation
        A2, cache = forward_propagation(X, parameters)
        
        # Calculate the cost
        cost = calc_cost(A2, Y, parameters)
        
        # Backward propagation
        gradients = backward_propagation(parameters, cache, X, Y)
        
        # Update parameters using gradiente descent
        parameters = update_parameters(parameters, gradients)
        
        # Print cost every 1000 iterations
        if print_cost and i % 1000 == 0:
            print("The cost after {0}: {1}".format(i, cost))
    
    return parameters
        
            
        
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    