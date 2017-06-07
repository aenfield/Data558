# Implements L2-regularized logistic regression using the fast gradient descent algorithm.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# for sklearn wrapper around this implementation
from sklearn.base import BaseEstimator, ClassifierMixin


# ---
# Gradient and objective functions
# ---

def compute_gradient_logistic_regression(beta, X, y, lam=1):
    """
    Evaluates the gradient of the L2-regularized logistic regression objective function.

    :param beta: a vector of coefficients, one per predictor
    :param X: a matrix with the predictor data, of size number of observations by number of predictors
    :param y: a vector of the actual values, one per observation
    :param lam: a scalar controlling the degree of regularization
    :return: a vector of updated coefficients, one per predictor
    """
    exp_term = np.exp(-y * (X.dot(beta)))
    scalar_term = (y * exp_term) / (1 + exp_term)
    penalty = 2*lam*beta
    grad_beta = (1/len(X)) * -scalar_term[np.newaxis].dot(X) + penalty
    return grad_beta.flatten() # flatten to return a vector instead of a 1-D array

def compute_objective_logistic_regression(beta, X, y, lam=5):
    """
    Evaluates the L2-regularized logistic regression objective function.

    :param beta: a vector of coefficients, one per predictor
    :param X: a matrix with the predictor data, of size number of observations by number of predictors
    :param y: a vector of the actual values, one per observation
    :param lam: a scalar controlling the degree of regularization
    :return: a scalar holding the objective value at this location/for this set of coefficients
    """
    obj = 1/len(X) * sum( np.log(1 + np.exp(-y * (X.dot(beta)))) )
    obj = obj + lam*(np.linalg.norm(beta))**2
    return obj


# ---
# Fast gradient descent
# ---

def fastgradalgo(X, y, t_init, grad_func, obj_func, iter=100, lam=1, t_func=None):
    """
    Implements the fast gradient descent algorithm. This simple implementation uses only the number of iterations as
    a stopping criteria.

    :param X: a matrix with the predictor data, of size number of observations by number of predictors
    :param y: a vector of the actual values, one per observation
    :param t_init: initial step size (if no function is passed as t_func, this step size is used throughout)
    :param grad_func: a reference to the function that calculates the gradient vector
    :param obj_func: a reference to the function that calculates the objective value
    :param iter: number of iterations to run before returning all coefficients
    :param lam: a scalar controlling the degree of regularization
    :param t_func: if set, called to determine the step size dynamically on each iteration
    :return: a matrix of coefficient values, one row per iteration and one column per feature in the passed X data
    """
    beta = np.zeros(X.shape[1])  # shape[1] is the number of columns
    theta = beta.copy()  # we init theta to zeros too
    t = t_init
    beta_vals = pd.DataFrame(beta[np.newaxis, :])

    for i in range(0, iter):
        if t_func:
            t = t_func(beta, X, y, grad_func, obj_func, t_init)

        beta_prev = beta.copy()
        beta = theta - (t * grad_func(theta, X, y, lam))
        theta = beta + (i / (i + 3)) * (beta - beta_prev)

        beta_vals = beta_vals.append(pd.DataFrame(beta[np.newaxis]), ignore_index=True)

    return beta_vals

def get_final_coefs(coef_vals):
    """
    Returns the final set of coefficients - these are the coefficents that typically will be used for prediction.

    :param coef_vals: a matrix of one or more sets of coefficients
    :return: A 1-D matrix containing a coefficient for each predictor
    """
    return coef_vals[-1:].values

def backtracking(coefs, X, y, grad_func, obj_func, t=1, alpha=0.5, beta=0.5, iter=100, lam=1):
    """
    Returns a value of t - the step size - for use with gradient descent, obtained via backtracking.

    :param coefs: current 'location' on the shape being minimized, represented by a vector of coefficients
    :param X: a matrix with the predictor data, of size number of observations by number of predictors
    :param y: a vector of the actual values, one per observation
    :param grad_func: a reference to the function that calculates the gradient vector
    :param obj_func: a reference to the function that calculates the objective value
    :param t: initial value of t, used by the backtracking algorithm
    :param alpha: alpha parameter to the backtracking algorithm
    :param beta: beta parameter to the backtracking algorithm
    :param iter: number of iterations to run before returning all coefficients
    :param lam: a scalar controlling the degree of regularization
    :return: the step size given this input
    """
    grad_coefs = grad_func(coefs, X, y, lam)
    norm_grad_coefs = np.linalg.norm(grad_coefs) # norm of gradient
    found_t = False
    iter = 0
    while (found_t == False and iter < iter):
        attempted_value = obj_func(coefs - (t * grad_coefs), X, y, lam)
        compare_value = obj_func(coefs, X, y, lam) - (alpha * t * norm_grad_coefs**2)
        if attempted_value < compare_value:
          found_t = True
        elif iter == iter:
            # TODO should check whether 'exit' is valid
            exit("Backtracking: max iterations reached")
        else:
            t = t * beta
            iter = iter + 1

    return t


# ---
# Metrics
# ---

def get_probability(logodds):
    """
    Returns the probability associated with the passed log odds value.

    :param logodds: a log odds value
    :return: the probability associated with this log odds value
    """
    return 1 / (1 + np.exp(-logodds))

def get_y_pred(coefs, X, prob_threshold=0.5):
    """
    Returns the predicted -1 or 1 for each row of data.

    :param coefs: coefficent values, one per predictor
    :param X: a matrix with the predictor data, of size number of observations by number of predictors
    :param prob_threshold: probabilities above this threshold are assigned 1, equal or below are assigned -1
    :return: a vector of predictions, one per observation in X
    """
    y_pred = X.dot(coefs.T).ravel()  # ravel to convert to vector

    # for logistic regression convert to a prob and use a prob threshold
    probs = get_probability(y_pred)
    y_thresholded = np.where(probs > prob_threshold, 1, -1)
    return y_thresholded


# ---
# Plots
# ---

def objective_plot(all_coefs, X, y,
                   obj_func = compute_objective_logistic_regression,
                   lam=1):
    """
    Plots the value of the objective function per iteration.

    :param all_coefs: a matrix of coefficient values, one row per iteration and one column per feature in the passed X data
    :param X: a matrix with the predictor data, of size number of observations by number of predictors
    :param y: a vector of the actual values, one per observation
    :param obj_func: a reference to the function that calculates the objective value
    :param lam: a scalar controlling the degree of regularization
    :return: nothing.
    """
    obj_values = all_coefs.apply(lambda r: obj_func(
        r.as_matrix(), X, y, lam=lam), axis=1)

    plt.subplot(1, 2, 1)
    plt.plot(all_coefs.index, obj_values)
    plt.xlabel("Iteration")
    plt.ylabel("Objective value")



# ---
# Sklearn-compliant estimator wrapper
#
# Wraps fastgradalgo and obj/grad functions in a class w/ fit and predict methods so this implementation can be
# used by other sklearn code that expects a classifier. Parts are based on a writeup at
# http://danielhnyk.cz/creating-your-own-estimator-scikit-learn/.
#
# This is a bonus/not required by the description; I haven't written documentation for each individual function -
# for more, see the reference above.
# ---

class MyLogisticRegression(BaseEstimator, ClassifierMixin):
    default_iters = 300
    default_t_init = 0.01

    def __init__(self, C=1, iter=default_iters):
        self.C = C
        self.iter = iter

    def fit(self, X, y, t_init=default_t_init):
        self.all_coefs_ = fastgradalgo(X, y, t_init,
            grad_func = compute_gradient_logistic_regression,
            obj_func = compute_objective_logistic_regression,
            lam=self.C, iter=self.iter)

        return self

    def predict(self, X, prob_threshold=0.5):
        self.raise_if_not_fit()

        probs = get_probability(self.decision_function(X))
        y_thresholded = np.where(probs > prob_threshold, 1, -1)
        return y_thresholded

    # seems like the convention w/ sklearn models is that decision_function returns the score, while
    # predict returns a class label - for now, I'll return the log odds (per additional investigation -
    # the docs for the OneVsRestClassifier metaclassifier - it looks like that classifier uses either
    # decision_function or predict_proba (the latter which gives the probabilities, I think)
    def decision_function(self, X):
        y_pred = X.dot(get_final_coefs(self.all_coefs_).T).ravel()  # ravel to convert to vector
        return y_pred

    def get_coefs(self):
        self.raise_if_not_fit()
        return get_final_coefs(self.all_coefs_)

    def raise_if_not_fit(self):
        if self.all_coefs_ is None:
            raise RuntimeError("Model has not yet been fit.")
