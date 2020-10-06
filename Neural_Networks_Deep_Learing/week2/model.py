# -*- coding: utf-8 -*-

from lr_utils import *


def lr_model(X_train, Y_train, X_test, Y_test, epochs=2000, learning_rate=0.005, print_cost=False):
    """
    Builds logistic regression model using functions defined in lr_utils


    Parameters
    ----------
    X_train : numpy array
        training set shape (num_px * num_px * 3, m_train).
    Y_train : numpy array
        training labels shape (1, m_train).
    X_test : numpy array 
        testing set shape (num_px * num_px * 3, m_train).
    Y_test : numpy array
        testign labels shape (1, m_train).
    epochs : int, optional
        hyperparameter used to fine tune algorithtm, iterations. The default is 2000.
    learning_rate : float, optional
        hyperparameter used in update rule. The default is 0.5.
    print_cost : Boolean, optional
        prints cost every 100 iterations. The default is False.

    Returns
    -------
    d : dictionary
        information about the model.
    """
    
    # Initialize the w, and b
    w, b = initialize_parameters(X_train.shape[0])
    
    # optimize parameters 
    parameters, gradients, costs = optimize_parameters(w, b, X_train, Y_train, epochs, learning_rate, print_cost)
    
    # Get optoimied w and b
    w = parameters["w"]
    b = parameters["b"]
    
    # Make predictions
    Y_pred_test = predict(w, b, X_test)
    Y_pred_train = predict(w, b, X_train)
    
    
    # Print test/train errors
    print("Train accuracy: {}%".format(100 - np.mean(np.abs(Y_pred_train - Y_train)) * 100))
    print("Test accuracy: {}%".format(100 - np.mean(np.abs(Y_pred_test - Y_test)) * 100))
    
    d = {
            "costs": costs,
            "Y_prediction_test": Y_pred_test, 
            "Y_prediction_train" : Y_pred_train, 
            "w" : w, 
            "b" : b,
            "learning_rate" : learning_rate,
            "epochs": epochs
        }
    
    return d
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
