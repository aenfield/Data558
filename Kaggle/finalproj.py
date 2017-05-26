import numpy as np

def compute_gradient_logistic_regression(beta, x, y, lam=5):
    """Compute the gradient function for logistic regression."""
    exp_term = np.exp(-y * (x.dot(beta)))
    scalar_term = (y * exp_term) / (1 + exp_term)
    penalty = 2*lam*beta
    grad_beta = (1/len(x)) * -scalar_term[np.newaxis].dot(x) + penalty
    return grad_beta.flatten() # flatten to return a vector instead
                               # of a 1-D array

def compute_objective_logistic_regression(beta, x, y, lam=5):
    """Compute the objective function for logistic regression."""
    obj = 1/len(x) * sum( np.log(1 + np.exp(-y * (x.dot(beta)))) )
    obj = obj + lam*(norm(beta))**2
    return obj