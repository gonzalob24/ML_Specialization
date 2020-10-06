#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Project Name: Logistic Regressor to classify cats
    
Project Description:
    Do not use loops (for/while) in your code, unless the instructions explicitly ask you to do so.
    
Files used: 
    

Lessons Learned:
    You will learn to:
    Build the general architecture of a learning algorithm, including:
    Initializing parameters
    Calculating the cost function and its gradient
    Using an optimization algorithm (gradient descent)
    Gather all three functions above into a main model function, in the right order.

@author: gonzalobetancourt

"""
import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage
from lr_utils import *
from model import lr_model
from skimage.transform import resize
import imageio

# Load the h5py data 
X_train, Y_train, X_test, Y_test, classes = load_dataset()

# Look at an image 
index = 25
plt.imshow(X_train[index])
plt.title("y = " + str(Y_train[:, index]) + " , it's a '" + classes[np.squeeze(Y_train[:, index])].decode("utf-8") + "' picture.")
# print("y = " + str(Y_train[:, index]) + " , it's a '" + classes[np.squeeze(Y_train[:, index])].decode("utf-8") + "' picture.")

# Explore dimension fo the data. Images of a (m, np_px, np_px, 3)
m_train = X_train.shape[0]
m_test = X_test.shape[0]
num_px = X_train.shape[1]

# Flatten images of (np_px, np_px, 3) --> (np_px * np_px * 3, 1)
# Each column represents a flatten image
# There are two ways of flattening an image
flatten_image = X_train.reshape(X_train.shape[1] * X_train.shape[2] * X_train.shape[3], X_train.shape[0])
flatten_image.shape

X_train_flatten = X_train.reshape(X_train.shape[0], -1).T
X_train_flatten.shape
X_test_flatten = X_test.reshape(X_test.shape[0], -1).T
X_test_flatten.shape

print ("train_set_x_flatten shape: " + str(X_train_flatten.shape))
print ("train_set_y shape: " + str(Y_train.shape))
print ("test_set_x_flatten shape: " + str(X_test_flatten.shape))
print ("test_set_y shape: " + str(Y_test.shape))
print ("sanity check after reshaping: " + str(X_train_flatten[0:5,0]))

# RGB color range 0-255
# Standardize the set by dividing by 255 instead of centering and standardizing 
# the data set --> subtract the mean and divide by std

X_train_std = X_train_flatten/255.0
X_test_std = X_test_flatten/255.0


d = lr_model(X_train_std, Y_train, X_test_std, Y_test, epochs=2000, learning_rate=0.005, print_cost=True)


costs = np.squeeze(d["costs"])
plt.plot(costs)
plt.ylabel('cost')
plt.xlabel('iterations (per hundreds)')
plt.title("Learning rate =" + str(d["learning_rate"]))
plt.show()


# Different learning rates
learning_rates = [0.2, 0.05, 0.005, 0.001, 0.0001]
models = {}

for lr in learning_rates:
    print("Learning Rate is: {}".format(lr))
    models[str(lr)] = lr_model(X_train_std, Y_train, X_test_std, Y_test, epochs=2000, learning_rate=lr, print_cost=False)
    print ('\n' + "-------------------------------------------------------" + '\n')

for i in learning_rates:
    plt.plot(np.squeeze(models[str(i)]["costs"]), label= str(models[str(i)]["learning_rate"]))

plt.ylabel('cost')
plt.xlabel('iterations (hundreds)')

legend = plt.legend(loc='upper center', shadow=True)
frame = legend.get_frame()
frame.set_facecolor('0.90')
plt.show()


# Test an image not in train or test set


my_image = "test_cat.jpg"
test_picture(my_image, d, classes)











