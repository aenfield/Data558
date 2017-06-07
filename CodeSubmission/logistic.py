# Implements logistic regression, per guidance given in class, using either a standard (slow) gradient
# descent algorithm or a (theoretically faster) fast gradient descent algorithm.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# for sklearn wrapper
from sklearn.base import BaseEstimator, ClassifierMixin


# ---
# Gradient and objective functions
# ---
def compute_gradient_logistic_regression(beta, x, y, lam=1):
    exp_term = np.exp(-y * (x.dot(beta)))
    scalar_term = (y * exp_term) / (1 + exp_term)
    penalty = 2*lam*beta
    grad_beta = (1/len(x)) * -scalar_term[np.newaxis].dot(x) + penalty
    return grad_beta.flatten() # flatten to return a vector instead of a 1-D array

def compute_objective_logistic_regression(beta, x, y, lam=5):
    obj = 1/len(x) * sum( np.log(1 + np.exp(-y * (x.dot(beta)))) )
    obj = obj + lam*(np.linalg.norm(beta))**2
    return obj


# ---
# Fast gradient descent
# ---

def fastgradalgo(x, y, t_init, grad_func, obj_func, max_iter=100, lam=1, t_func=None):
    """
    Implements the fast gradient descent algorithm. Uses t_func
    to determine t, or uses a fixed size of t_init if
    t_func is None. That is, pass in a function like
    'backtracking' to do a line search.
    """
    beta = np.zeros(x.shape[1])  # shape[1] is the number of columns
    theta = beta.copy()  # we init theta to zeros too
    t = t_init
    beta_vals = pd.DataFrame(beta[np.newaxis, :])

    for i in range(0, max_iter):
        if t_func:
            t = t_func(beta, x, y, grad_func, obj_func, t_init)

        beta_prev = beta.copy()
        beta = theta - (t * grad_func(theta, x, y, lam))
        theta = beta + (i / (i + 3)) * (beta - beta_prev)

        beta_vals = beta_vals.append(pd.DataFrame(beta[np.newaxis]), ignore_index=True)

    return beta_vals

def get_final_coefs(beta_vals):
    return beta_vals[-1:].values

def backtracking(coefs, x, y, grad_func, obj_func, t=1, alpha=0.5, beta=0.5, max_iter=100, lam=1):
    """
    Returns a value of t, for use w/ gradient descent, obtained via backtracking."""
    grad_coefs = grad_func(coefs, x, y, lam)
    norm_grad_coefs = np.linalg.norm(grad_coefs) # norm of gradient
    found_t = False
    iter = 0
    while (found_t == False and iter < max_iter):
        attempted_value = obj_func(coefs - (t * grad_coefs), x, y, lam)
        compare_value = obj_func(coefs, x, y, lam) - (alpha * t * norm_grad_coefs**2)
        if attempted_value < compare_value:
          found_t = True
        elif iter == max_iter:
            # TODO should ideally check whether 'exit' is valid
            exit("Backtracking: max iterations reached")
        else:
            t = t * beta
            iter = iter + 1

    return t
    #return(t, iter)


# ---
# Wrap fastgradalgo and obj/grad functions in a class w/ fit and predict methods
# (No tests, for now :-( )
# ---

# i implemented fit and predict and it didn't work, so I added a bit from
# http://danielhnyk.cz/creating-your-own-estimator-scikit-learn/

class MyLogisticRegression(BaseEstimator, ClassifierMixin):
    default_max_iters = 300
    default_t_init = 0.01

    def __init__(self, C=1, max_iter=default_max_iters):
        self.C = C
        self.max_iter = max_iter

    def fit(self, X, y, t_init=default_t_init):
        self.all_coefs_ = fastgradalgo(X, y, t_init,
            grad_func = compute_gradient_logistic_regression,
            obj_func = compute_objective_logistic_regression,
            lam=self.C, max_iter=self.max_iter)

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

# ---
# Metrics
# ---

def get_probability(logodds):
    return 1 / (1 + np.exp(-logodds))

def get_y_pred(coefs, X, prob_threshold=0.5):
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
    obj_values = all_coefs.apply(lambda r: obj_func(
        r.as_matrix(), X, y, lam=lam), axis=1)
    # grad_norm_values = all_coefs.apply(lambda r: np.linalg.norm(grad_func(
    #     r.as_matrix(), X, y, lam=lam)), axis=1)

    plt.subplot(1, 2, 1)
    plt.plot(all_coefs.index, obj_values)
    plt.xlabel("Iteration")
    plt.ylabel("Objective value")

    # plt.subplot(1, 2, 2)
    # plt.plot(all_coefs.index, grad_norm_values)
    # plt.xlabel("Iteration")
    # plt.ylabel("Norm of gradient")