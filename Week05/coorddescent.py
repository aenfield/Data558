#from __future__ import division
#import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
#from sklearn.preprocessing import StandardScaler
#from sklearn.linear_model import LassoCV


# I'm using Corinne's implementation here because - while I actually think it's at least somewhat possibly my
# implementation from last week might be accurate based on a very quick comparison to the posted solution - I'm not
# sure of it. I'd rather save all the time possible for new stuff than try to reimplement things that aren't the
# immediate focus and take away time from higher priority learning.


# Part (a)
def soft_threshold(c, lambduh):
    if c < -lambduh:
        return c+lambduh
    elif c > lambduh:
        return c-lambduh
    else:
        return 0


def coord_descent_solution1d(x, y, lambduh):
    c = 2*x*(y-0)
    a = 2*x**2
    return soft_threshold(c, lambduh)/a


# Part (b)
def coord_descent_solution(j, X, lambduh, beta, a, xy):
    """
    Compute the solution of the coordinate descent problem at one iteration
    :param j: Coordinate to optimize over
    :param X: Matrix of predictor values
    :param lambduh: Regularization parameter
    :param a: Precomputed values a_1,...,a_d
    :param xy: Precomputed values \sum x_{i1}y_i, ..., \sum x_{id}y_i
    :return: beta_j: Updated value of beta_j
    """
    # Method 1:
    # n, d = X.shape
    # c_j = 0
    # for i in range(n):
    #     beta_x = 0
    #     for k in range(0, d):
    #         if k != j:
    #             beta_x += beta[k]*X[i, k]
    #     c_j += 2*X[i, j]*(y[i] - beta_x)
    # Method 2:
    c_j = 2*(xy[j] - np.sum(X[:,j]*(np.inner(X, beta) - X[:, j]*beta[j])))
    # Method 3:
    # beta_minus_j = np.concatenate((beta[0:j], beta[j+1:]))
    # x_minus_j = np.hstack((X[:, 0:j], X[:, j+1:]))
    # c_j = 2*(xy[j] - np.sum(X[:,j]*np.dot(x_minus_j, beta_minus_j[:, np.newaxis]).T))

    return soft_threshold(c_j, lambduh)/a[j]


def cycliccoorddescent(beta, X, y, lambduh, max_iter):
    d = np.size(X, 1)
    a = 2*np.sum(X**2, axis=0)
    xy = np.dot(X.T, y)
    all_betas = beta
    for i in range(max_iter):
        for j in range(d):
            beta[j] = coord_descent_solution(j, X, lambduh, beta, a, xy)
        all_betas = np.vstack((all_betas, beta))

    return all_betas


# Part (f): Function to randomly select coordinate
def pickcoord(d):
    return np.random.randint(d)


# Part (g): Random coordinate descent
def randcoorddescent(beta, X, y, lambduh, max_iter):
    d = np.size(X, 1)
    a = 2*np.sum(X**2, axis=0)
    xy = np.dot(X.T, y)
    all_betas = beta
    for i in range(max_iter*d):
        j = pickcoord(d)
        beta[j] = coord_descent_solution(j, X, lambduh, beta, a, xy)
        if i % d == 0:
            all_betas = np.vstack((all_betas, beta))

    return all_betas
