import numpy as np


#activation functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
    
def dsigmoid(x):
    return sigmoid(x=x) * (1 - sigmoid(x=x))
    
def relu(x):
    return np.maximum(0.0, x)
    
def drelu(x):
    return np.where(x > 0, 1, 0)
    
def tanh(x):
    return np.tanh(x)


def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))  # Subtracting max for numerical stability
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    
def dtanh(x):
    return 1 - np.square(np.tanh(x))


#loss functions
def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def binary_cross_entropy(y_true, y_pred):
    epsilon = 1e-15 
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))





#dicts
loss_functions : dict = {"mse": mean_squared_error,
                         "cross-entropy": binary_cross_entropy
                         }

activation_functions: dict = {"sigmoid": sigmoid,
                              "dsigmoid": dsigmoid,
                              "relu": relu,
                              "drelu": drelu,
                              "tanh": tanh,
                              "dtanh": dtanh,
                              "softmax": softmax
                              }