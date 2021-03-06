# -*- coding: utf-8 -*-

"""
Project Name: Deep Learning Model for Image Classification
    
Project Description: Building a Deep Learning Model From Scratch
    
Files used: utility functions

@author: gonzalobetancourt

"""

import time
import numpy as np
import h5py

import matplotlib.pyplot as plt
import scipy
from PIL import Image
from utility_functions import *

# default size of plots
plt.rcParams["figure.figsize"] = (5.0, 4.0)
plt.rcParams["image.interpolation"] = "nearest"
plt.rcParams["image.cmap"] = "gray"

np.random.seed(1)

# load the data
train_x_orig, train_y, test_x_orig, test_y, classes = load_data()

# See of a picture
index = 10
plt.imshow(train_x_orig[index])
print ("y = " + str(train_y[0,index]) + ". It's a " + classes[train_y[0,index]].decode("utf-8") +  " picture.")

# Explore the dataset -- shape size
m_train = train_x_orig.shape[0]
num_px = train_x_orig.shape[1]
m_test = test_x_orig.shape[0]

print ("Number of training examples: " + str(m_train))
print ("Number of testing examples: " + str(m_test))
print ("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
print ("train_x_orig shape: " + str(train_x_orig.shape))
print ("train_y shape: " + str(train_y.shape))
print ("test_x_orig shape: " + str(test_x_orig.shape))
print ("test_y shape: " + str(test_y.shape))

# Reshape training and test examples (num_xp * num_xp * 3, m)
train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T   # The "-1" makes reshape flatten the remaining dimensions
test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T

# Standardize data to have feature values between 0 and 1.
train_x = train_x_flatten/255.
test_x = test_x_flatten/255.

print ("train_x's shape: " + str(train_x.shape))
print ("test_x's shape: " + str(test_x.shape))


# First I will build a two layer NN and then gneralize it to allo L-hidden layers

# Set the dimenssions
n_x = 12288     # num_px * num_px * 3
n_h = 7
n_y = 1
layers_dims = (n_x, n_h, n_y)

def two_layer_NN(X, Y, layers_dims, learning_rate=0.0075, iterations=3000, print_cost=False):
    """
    Implements a two-layer neural network: Linear->RELU->Linear->SIGMOID.


    Parameters
    ----------
    X : numpy array
        input data, of shape (n_x, number of examples).
    Y : numpy array
        true "labels" vector (containing 1 if cat, 0 if non-cat), of shape (1, number of examples).
    layers_dims : list
        dimensions of the layers (n_x, n_h, n_y).
    learning_rate : float, optional
        learning rate of the gradient descent update rule. The default is 0.0075.
    iterations : int, optional
        number of iterations of the optimization loop. The default is 3000.
    print_cost : Boolean, optional
        If set to True, this will print the cost every 100 iterations . The default is False.

    Returns
    -------
    Parameters : dictionary.
        contains W1, W2, b1, and b2
    """
    
    np.random.seed(1)
    grads = {}
    costs = []
    m = X.shape[1]
    
    (n_x, n_h, n_y) = layers_dims
    
    # Initialize parameters dictionary
    parameters = initialize_parameters(n_x, n_h, n_y)
    
    # Get parameters from dictionary
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    
    # Calc gradient descent with loop
    for i in range(0, iterations):
        # 1) Forward propagation: Linear -> RELU -> Linear -> SIGMOID. 
        # Inputs: "X, W1, b1, W2, b2". Output: "A1, cache1, A2, cache2".
        A1, cache1 = linear_activation_forward(X, W1, b1, 'relu')
        A2, cache2 = linear_activation_forward(A1, W2, b2, 'sigmoid')

        # 2) Compute cost
        cost = compute_cost(A2, Y)
        
        # 3) Initializing backward propagation
        dA2 = - (np.divide(Y, A2) - np.divide(1 - Y, 1 - A2))
        
        # Backward propagation. 
        # Inputs: "dA2, cache2, cache1". Outputs: "dA1, dW2, db2; also dA0 (not used), dW1, db1".
        dA1, dW2, db2 = linear_activation_backward(dA2, cache2, 'sigmoid')
        dA0, dW1, db1 = linear_activation_backward(dA1, cache1, 'relu')
        
        # 4) Set grads['dWl'] to dW1, grads['db1'] to db1, grads['dW2'] to dW2, grads['db2'] to db2
        grads['dW1'] = dW1
        grads['db1'] = db1
        grads['dW2'] = dW2
        grads['db2'] = db2
        
        # 5) Update parameters.
        parameters = update_parameters(parameters, grads, learning_rate)
        
        # Retrieve W1, b1, W2, b2 from parameters
        W1 = parameters["W1"]
        b1 = parameters["b1"]
        W2 = parameters["W2"]
        b2 = parameters["b2"]
        
        # Print the cost every 100 training example
        if print_cost and i % 100 == 0:
            print("Cost after iteration {}: {}".format(i, np.squeeze(cost)))
        if print_cost and i % 100 == 0:
            costs.append(cost)
    
    # plot the cost
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('Iterations (per hundreds)')
    plt.title("Learning rate = " + str(learning_rate))
    plt.show()
    
    return parameters

# L layer NN 
def L_layer_model(X, Y, layers_dims, learning_rate = 0.0075, iterations = 3000, print_cost=False, lmbd=0, keep_prob=1):#lr was 0.009
    """
    Implements a L-layer neural network: Linear->RELU*(L-1)->Linear->SIGMOID.
    
    Parameters:
    X : numpy array.
        data, numpy array of shape (num_px * num_px * 3, number of examples)
    Y : numpy array.
        true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
    layers_dims : list 
        contains the input size and each layer size, of length (number of layers + 1).
    learning_rate : float.
        learning rate of the gradient descent update rule
    num_iterations : int. 
        number of iterations of the optimization loop
    print_cost : Boolean.
        if True, it prints the cost every 100 steps
    
    Returns:
    parameters : dictionary
        parameters learnt by the model. They can then be used to predict.
    """

    np.random.seed(1)
    costs = []                         # keep track of cost
    
    # Initialize Parameters
    parameters = initialize_parameters_deep(layers_dims)
    
    # Loop for gradient descent
    for i in range(0, iterations):

        # 1) Forward propagation: Linear -> RELU*(L-1) -> Linear -> SIGMOID
        AL, caches = deep_model_forward(X, parameters)
        
        # 2) Compute cost.
        cost = compute_cost(AL, Y)
    
        # 3) Backward propagation
        grads = deep_model_backward(AL, Y, caches)
 
        # 4) Update parameters
        parameters = update_parameters(parameters, grads, learning_rate)
                
        # Print the cost every 100 training example
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
        if print_cost and i % 100 == 0:
            costs.append(cost)
            
    # plot the cost
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    
    return parameters
        
parameters_2_layers = two_layer_NN(train_x, train_y, layers_dims = (n_x, n_h, n_y), iterations = 2500, print_cost=True)
predictions_train_2_layers = predict(train_x, train_y, parameters_2_layers)
pred_test_2_layers = predict(test_x, test_y, parameters_2_layers)

layers_dims = [12288, 20, 7, 5, 1]
parameters_L = L_layer_model(train_x, train_y, layers_dims, iterations = 2500, print_cost = True)
pred_train_L = predict(train_x, train_y, parameters_L)
pred_test_L = predict(test_x, test_y, parameters_L)


# Test another image
# change this to the name of your image file
my_image = "my_image.jpg" 

# the true class of your image (1 -> cat, 0 -> non-cat)
my_label_y = [1] 

fname = "images/" + my_image
# resize it first
image = np.array(Image.open(fname).resize((num_px, num_px)))
my_image = np.array(image).reshape((num_px*num_px*3,1))
my_image = my_image/255.
my_predicted_image = predict(my_image, my_label_y, parameters_L)

plt.imshow(image)
print ("y = " + str(np.squeeze(my_predicted_image)) + ", your L-layer model predicts a \"" + classes[int(np.squeeze(my_predicted_image)),].decode("utf-8") +  "\" picture.")

