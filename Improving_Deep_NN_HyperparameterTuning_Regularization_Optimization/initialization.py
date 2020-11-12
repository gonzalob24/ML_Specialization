import numpy as np

def initialize_parameters_zerto(layers_dims):
    """
    Initialize all parameters to zero. This is not the best way to initialize 
    parameters because the networl fails to break symmetry. Meaning that at 
    each layer the model is learning the exact same thing. 
    
    Weights should be initialized randomly to break symmetry

    Parameters
    ----------
    layer_dims : Array
        that contains the size of each layer.

    Returns
    -------
    parameters : python dictionary containing the parameters "W1", "b1", ..., "WL", "bL"
        W1 -- weight matrix of shape (layers_dims[1], layers_dims[0])
        b1 -- bias vector of shape (layers_dims[1], 1)
        ...
        WL -- weight matrix of shape (layers_dims[L], layers_dims[L-1])
        bL -- bias vector of shape (layers_dims[L], 1)

    """
    
    parameters = {}
    L = len(layers_dims)
    for layer in range(1, L):
        parameters["W" + str(layer)] = np.zeros((layers_dims[layer], layers_dims[layer - 1]))
        parameters["b" + str(layer)] = np.zeros((layers_dims[layer], 1))
        
    return parameters


def initialize_parameters_random(layers_dims):
    """
    To break symmetry in the network the weights should be initialized randomly.
    When using random initialization each neuron learns a different function of 
    its inputs. However, the size of the random initialization can have an 
    effect on the results. 
    
    If using large values to initialize the weights, the last activation 
    function are very close to 0 or 1 and when the model gets that example
    wrong the cost will be very high. 
    
    Poor initialization can lead to vanishing or exploding gradients which in 
    turn slows down optimization algorthim. You can train longer to get much 
    better results but the optimization will be slower. 

    Parameters
    ----------
    layer_dims : Array
        that contains the size of each layer.

    Returns
    -------
    parameters : python dictionary containing the parameters "W1", "b1", ..., "WL", "bL"
        W1 -- weight matrix of shape (layers_dims[1], layers_dims[0])
        b1 -- bias vector of shape (layers_dims[1], 1)
        ...
        WL -- weight matrix of shape (layers_dims[L], layers_dims[L-1])
        bL -- bias vector of shape (layers_dims[L], 1)
    """
    
    parameters = {}
    L = len(layers_dims)
    
    for layer in range(1, L):
        parameters["W" + str(layer)] = np.random.randn(layers_dims[layer], layers_dims[layer - 1]) * 10
        parameters["b" + str(layer)] = np.zeros((layers_dims[layer], 1))
    
    

def initialization_parameters_he(layers_dims):
    """
    Initializing the weights to small random numbers is much better. Use
    He initialization with a Relu activation.

    Parameters
    ----------
    layer_dims : Array
        that contains the size of each layer.

    Returns
    -------
    parameters : python dictionary containing the parameters "W1", "b1", ..., "WL", "bL"
        W1 -- weight matrix of shape (layers_dims[1], layers_dims[0])
        b1 -- bias vector of shape (layers_dims[1], 1)
        ...
        WL -- weight matrix of shape (layers_dims[L], layers_dims[L-1])
        bL -- bias vector of shape (layers_dims[L], 1)
    """
    
    parameters = {}
    L = len(layers_dims)
    
    for layer in range(1, L):
        parameters["W" + str(layer)] = np.random.randn(layers_dims[layer], layers_dims[layer - 1]) * np.sqrt(2/layers_dims[layer - 1])
        parameters["b" + str(layer)] = np.zeros((layers_dims[layer], 1))
    
    
    
    
    
    
    
    
        