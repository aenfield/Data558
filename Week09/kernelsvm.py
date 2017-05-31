import numpy as np
from sklearn.metrics.pairwise import linear_kernel, polynomial_kernel

def computegram_linear(X):
    """
    Return the kernel matrix for X, using a linear kernel    
    """
    return X.dot(X.T)
    # we should/could use the canned implementation, but we'll do this manually to show we can
    #return linear_kernel(X)

def computegram_polynomial(X, degree, coef0):
    """
    Return the kernel matrix for X, using a polynomnial kernel
    """
    return (X.dot(X.T) + coef0)**degree
    # we should/could use the canned implementation, but we'll do this manually to show we can
    #return linear_kernel(X)

def kerneleval_linear(X, new_observation):
    """
    Evaluate new_observation compared to the existing observations, using a linear kernel
    """
    return new_observation.dot(X.T)

def compute_kernelsvm_gradient():
    return 0

def compute_kernelsvm_objective():
    return 0