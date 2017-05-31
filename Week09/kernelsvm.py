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

def compute_kernelsvm_gradient(alphas, K, y, lam):
    grad_beta_max_term = np.maximum(0, 1 - (y * (K.dot(np.array(alphas)))))[np.newaxis].T
    grad_beta_sum_term = y[np.newaxis].T * K * grad_beta_max_term
    grad_beta_without_penalty = (-2 / len(y)) * np.sum(grad_beta_sum_term, 0)
    grad_beta_penalty = 2 * lam * (K.dot(alphas))
    return grad_beta_without_penalty + grad_beta_penalty

def compute_kernelsvm_objective(alphas, K, y, lam):
    objective_max_term = np.maximum(0, 1 - (y * (K.dot(np.array(alphas)))))[np.newaxis].T
    obj_without_penalty = (1 / len(y)) * np.sum(objective_max_term**2)
    obj_penalty = lam * alphas * (K.dot(alphas))
    return obj_without_penalty + obj_penalty
