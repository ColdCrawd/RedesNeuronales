import numpy as np

#Step Function
def step_function(x):
    return np.where(x>=0, 1, 0)

#Sigmoide
def sigmoid(x):
    return 1/(1+np.exp(-x))

#Tanh
def tanh(x):
    return np.tanh(x)

#Relu
def relu(x):
    return np.maximum(0, x)

#Leaky_Relu
def leaky_relu(x, alpha = 0.01):
    return np.where(x>=0, x, alpha*x)
    

#Telu
def telu(x, alpha = 0.01):
    return np.where(x>=0, x, alpha*(np.exp(x)-1))

#Softmax
def softmax(x):
    exp = np.exp(x-np.max(x, axis=1, keepdims=True))
    return exp/np.sum(exp, axis=1, keepdims=True)
