# -*- coding: utf-8 -*-
"""
Use regularization when the model is overfitting. The model does well with the
training set but fails to give good results but fails to generalize well
to new instnaces or data that it has not seen before. 

Two methods to keep in  mind

L2 regularization: compute_cost_with_regularization() and backward_propagation_with_regularization()
Dropout: forward_propagation_with_dropout() and backward_propagation_with_dropout()

Will work on implementing these fucntions to the deep _model from week 4 in 
perv course.

"""

def add_this_to_week4_deep_model(lmbd=0, keep_prob=1):
    """
    
    lambd:int
        regularization hyperparameter, scalar
    keep_prob: float
        probability of keeping a neuron active during drop-out, scalar.
    
    """
    
    grads = {}
    costs = []                            # to keep track of the cost
    m = X.shape[1]                        # number of examples
    layers_dims = [X.shape[0], 20, 3, 1]
    
    # Initialize parameters dictionary.
    parameters = initialize_parameters(layers_dims)
    
    # Forward propagation: LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SIGMOID.
    if keep_prob == 1:
        A3, cache = forward_propagation(X, parameters)
    elif keep_prob < 1:
        A3, cache = forward_propagation_with_dropout(X, parameters, keep_prob)
        
    # Cost function
    if lambd == 0:
        cost = compute_cost(a3, Y)
    else:
        cost = compute_cost_with_regularization(a3, Y, parameters, lambd)
        
    # Backward propagation.
    assert(lambd==0 or keep_prob==1)    # it is possible to use both L2 regularization and dropout, 
                                        # but this assignment will only explore one at a time
    if lambd == 0 and keep_prob == 1:
        grads = backward_propagation(X, Y, cache)
    elif lambd != 0:
        grads = backward_propagation_with_regularization(X, Y, cache, lambd)
    elif keep_prob < 1:
        grads = backward_propagation_with_dropout(X, Y, cache, keep_prob)
        
        
    return parameters

"""
If lmbda is too large it is possible to oversmooth the decision boundary, resulting
with a model that has high bias. L2-Reg relies on the assumption that a model with
small weights is simpler than a model with large weights. So, it penalizes the 
square values of the weights in the cost function and make the weights smaller. 
It is costly for the cost to have large weights. --> Weight decay --> weights
are pushed to smaller values 
"""
        
        
def compute_cost_with_regularization(A3, Y, parameters, lambd):
    """
    The cost function with L2 regularization. See formula in week 1
    
    Arguments:
    A3: numpy matrix
        post-activation, output of forward propagation
        shape (output size, number of examples).
    Y: numpy array 
        true labels vector, of shape (output size, number of examples).
    parameters: dictionary
        containing parameters of the model.
    
    Returns:
    cost: float
        value of the regularized loss function formula
    """
    m = Y.shape[1]
    W1 = parameters["W1"]
    W2 = parameters["W2"]
    W3 = parameters["W3"]
    
    cross_entropy_cost = compute_cost(A3, Y) # This gives you the cross-entropy part of the cost
    
    ### START CODE HERE ### (approx. 1 line)
    L2_regularization_cost = (np.sum(np.square(W1)) + np.sum(np.square(W2)) + np.sum(np.square(W3))) * lambd / (2*m)
    ### END CODER HERE ###
    
    cost = cross_entropy_cost + L2_regularization_cost
    
    return cost
        
        
def backward_propagation_with_regularization(X, Y, cache, lambd):
    """
    The backward propagation of the baseline model to which we added an L2 regularization.
    
    Arguments:
    X : numpy matrix
        input dataset, of shape (input size, number of examples)
    Y : numpy vector
        true labels vector, of shape (output size, number of examples)
    cache :
        cache output from forward_propagation()
    lambd : int
        regularization hyperparameter, scalar
    
    Returns:
    gradients : dictionary
        With the gradients with respect to each parameter, activation and pre-activation variables
    """
    
    m = X.shape[1]
    (Z1, A1, W1, b1, Z2, A2, W2, b2, Z3, A3, W3, b3) = cache
    
    dZ3 = A3 - Y
    
    ### START CODE HERE ### (approx. 1 line)
    dW3 = 1./m * np.dot(dZ3, A2.T) + np.multiply((lambd/m), W3)
    ### END CODE HERE ###
    db3 = 1./m * np.sum(dZ3, axis=1, keepdims = True)
    
    dA2 = np.dot(W3.T, dZ3)
    dZ2 = np.multiply(dA2, np.int64(A2 > 0))
    ### START CODE HERE ### (approx. 1 line)
    dW2 = 1./m * np.dot(dZ2, A1.T) + np.multiply((lambd/m), W2)
    ### END CODE HERE ###
    db2 = 1./m * np.sum(dZ2, axis=1, keepdims = True)
    
    dA1 = np.dot(W2.T, dZ2)
    dZ1 = np.multiply(dA1, np.int64(A1 > 0))
    ### START CODE HERE ### (approx. 1 line)
    dW1 = 1./m * np.dot(dZ1, X.T) + np.multiply((lambd/m), W1)
    ### END CODE HERE ###
    db1 = 1./m * np.sum(dZ1, axis=1, keepdims = True)
    
    gradients = {"dZ3": dZ3, "dW3": dW3, "db3": db3,"dA2": dA2,
                 "dZ2": dZ2, "dW2": dW2, "db2": db2, "dA1": dA1, 
                 "dZ1": dZ1, "dW1": dW1, "db1": db1}
    
    return gradients
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    