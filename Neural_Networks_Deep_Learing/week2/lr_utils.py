import numpy as np
import h5py
from skimage.transform import resize
import imageio
import matplotlib.pyplot as plt

    
    
def load_dataset():
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

def sigmoid(z):
    """
    Compute the sigmid of z

    Parameters
    ----------
    z : Scalar numpy array
        It can be any size.

    Returns
    -------
    s : sigmoid of z

    """
    s = 1 / (1 + np.exp(-z))
    
    return s

def initialize_parameters(dim):
    """
    Creates a vector of zeros of shape (dim, 1) for w and initialize b to 0

    Parameters
    ----------
    dims : int
        size of the W vector.

    Returns
    -------
    w : vector of shape (dim, 1)
    b : bias set to 0

    """
    
    w = np.zeros((dim, 1))
    b = 0
    
    return w, b
    
    
    
def forward_backward_propagation(w, b, X, Y):
    """
    Implements the cost function and its gradients using partial; derivatives

    Parameters
    ----------
    w : numpy array
        size (num_px * num_px * 3, 1).
    b : int
        bias scalar.
    X : Numpy features 
        size (num_px * num_px * 3, m_training examples).
    Y : True labels
        size (1, m_samples).

    Returns
    -------
    costs : negative log-likelihood for logistic regression
    dw : gradient of the loss with respect to the weights 
    db : gradient of the loss with respect to the bias
    """
    
    m = X.shape[1]
    
    # Forward propagation
    A = sigmoid(np.dot(w.T, X) + b)
    cost = -1/m * np.sum(np.multiply(Y, np.log(A)) + np.multiply((1 - Y), np.log(1 - A)))
    
    
    # Backward propagation
    dw = 1/m * np.dot(X, (A - Y).T)
    db = 1/m * np.sum(A - Y)
    
    assert(dw.shape == w.shape)
    cost = np.squeeze(cost)
    assert(cost.shape == ())
    
    gradients = {"dw": dw,
                 "db": db}
    
    return gradients, cost

def optimize_parameters(w, b, X, Y, epochs, learning_rate, print_cost = False):
    """
    Helps optimize the parameters w and b using gradient descent algorithm
    

    Parameters
    ----------
    w : numpy array
        size (num_px * num_px * 3, 1).
    b : int
        bias scalar.
    X : Numpy features 
        size (num_px * num_px * 3, m_training examples).
    Y : True labels
        size (1, m_samples).
    epochs : int
        number of iterations.
    learning_rate : float
        learnign rate used for gradient descent.
    print_cost : Boolean, optional
        Prints the cost computed during optimizationm. The default is False.

    Returns
    -------
    prameters : dictionay containing w and b
    gradients : dictionary containing dw and db
    costs     : list of all costs during optimization

    """
    
    # Initialize empty list
    costs = []
    
    for i in range(epochs):
        # calc gradient and cost
        gradients, cost = forward_backward_propagation(w, b, X, Y)
        
        # get each derivative
        dw = gradients["dw"]
        db = gradients["db"]
        
        # update the parameters
        w = w - learning_rate * dw
        b = b - learning_rate * db
        
        # Keep track of the costs
        if i % 100 == 0:
            costs.append(cost)
        
        
        # print the cost every 100 iterations
        if print_cost and i % 100 == 0:
            print("The cost after iteration {}: {}".format(i, cost))
        
    parameters = {"w": w,
                  "b": b}
    
    gradients = {"dw": dw,
                 "db": db}
        
    return parameters, gradients, costs
    
    
def predict(w, b, X):
    """
    Make predictions using optimized parameters w, b

    Parameters
    ----------
    w : numpy array
        size (num_px * num_px * 3, 1).
    b : int
        bias scalar.
    X : Numpy features 
        size (num_px * num_px * 3, m_training examples).

    Returns
    -------
    Y-prediction : numpy array
        vector containing all predictions 0 or 1 for the examples in X

    """
    
    # size of the sample
    m = X.shape[1]
    
    # initialize Y_prediciton to empty np array
    Y_prediciton = np.zeros((1, m))
    w = w.reshape(X.shape[0], 1)
    
    # Vector A predicting the probabilities of a cat being present in the picture
    A = sigmoid(np.dot(w.T, X) + b)
    
    
    for i in range(A.shape[1]):
        # Probabilites to predictions
        if A[0,i] <= 0.5:
            Y_prediciton[0,i] = 0
        else:
            Y_prediciton[0,i] = 1
    
    return Y_prediciton


def test_picture(img, d, classes):
    """
    Tests any single picture to see if it is a cat
    
     Parameters
    ----------
    img : image
        any image of a cat.

    Returns
    -------
    None.

    """
       
    # We preprocess the image to fit your algorithm.
    fname = "images/" + img
    image = np.array(imageio.imread(fname, as_gray=False))
    image = image/255.0
    num_px = 64
    my_image = resize(image, (num_px,num_px), anti_aliasing=True).reshape((1, num_px*num_px*3)).T
    # my_image = scipy.misc.imresize(image, size=(num_px,num_px)).reshape((1, num_px*num_px*3)).T
    my_predicted_image = predict(d["w"], d["b"], my_image)
    plt.imshow(image)
    plt.title("y = " + str(np.squeeze(my_predicted_image)) + ", your algorithm predicts a \"" + classes[int(np.squeeze(my_predicted_image)),].decode("utf-8") +  "\" picture.")
    
    # print("y = " + str(np.squeeze(my_predicted_image)) + ", your algorithm predicts a \"" + classes[int(np.squeeze(my_predicted_image)),].decode("utf-8") +  "\" picture.")
    
    
if __name__ == "__main__":
    pass
    
    
    
    
    
    
    
    
    