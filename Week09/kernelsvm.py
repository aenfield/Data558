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
