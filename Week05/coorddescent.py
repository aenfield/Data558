import numpy as np
import pandas as pd

# I've included both my implementation and Corinne's implementation here, and my notebook uses both to compare
# and contrast. Given more time I would have done further work to learn check my code against Corinne's and to
# resolve any differences - this given the prof's comment on Wednesday night that he'd like us to update our own
# implementations with learnings. Coming just a few days before the homework was due (and given that I'd already
# finished the majority of the assignment because I knew I had family stuff on Thursday and Friday), I'm going to
# turn this in as is for now.

# My implementation from last week

# this is the equivalent of my computeobj function
def lasso_objective(beta, x, y, lam):
    obj = (1 / len(x)) * np.sum((y - x.dot(beta))**2)
    obj = obj + (lam * (np.sum(np.abs(beta))))
    return obj


def c_term(beta, x, y, j, lam):
    # list of items to keep, with item j removed
    idxs_to_keep = list(range(x.shape[1]))
    idxs_to_keep.remove(j-1)

    beta_minus_j = beta[idxs_to_keep]
    x_minus_j = x[:, idxs_to_keep]

    return 2 * np.sum(x[:, j-1] * (y - x_minus_j.dot(beta_minus_j)))


def a_term(x, j):
    return 2 * np.sum(x[:, j-1]**2)

# computes the formula of the solution of the lasso minimization problem
def minimize_beta_term(beta, x, y, j, lam):
    c = c_term(beta, x, y, j, lam)

    if c > lam:
        return (c - lam) / a_term(x, j)
    elif c < -lam:
        return (c + lam) / a_term(x, j)
    else:
        # c is between -lambda and +lambda
        return 0


# this is the equivalent of my pickcoord function
def get_sequence_of_js(feature_count, iterations, random=False):
    if not random:
        # get a sequence of feature indices/j values of size max_iter
        # we repeat enough times to have at least the right size or more,
        # then filter to just the right amount
        repeat_features_count = (iterations // feature_count) + 1
        sequence_of_js = np.repeat(np.arange(1, feature_count+1)[np.newaxis],
                                   repeat_features_count, 0).flatten()[:iterations]
    else:
        sequence_of_js = np.random.choice(np.arange(1, feature_count+1), iterations)

    return sequence_of_js


def coorddescent(x, y, lam, j_sequence):
    """
    Implements coordinate descent, using the provided sequence, which defines
    whether the params are handled cyclically or randomly
    """
    feature_count = x.shape[1]
    beta = np.zeros(feature_count)
    saved_beta_vals = pd.DataFrame(beta[np.newaxis, :])

    for j in j_sequence:
        b_hat_for_j = minimize_beta_term(beta, x, y, j, lam)
        beta_new_val = beta.copy()  # modify copy so we don't retroactively modify rows because of by reference
        beta_new_val[j-1] = b_hat_for_j # j indices are 1-based, so subtract one to set correct coef
        saved_beta_vals = saved_beta_vals.append(pd.DataFrame(beta_new_val[np.newaxis]), ignore_index=True)
        beta = beta_new_val

    return saved_beta_vals


def andrew_cycliccoorddescent(x, y, lam, max_iter=500):
    return coorddescent(x, y, lam, get_sequence_of_js(x.shape[1], max_iter))

def andrew_randcoorddescent(x, y, lam, max_iter=500):
    return coorddescent(x, y, lam, get_sequence_of_js(x.shape[1], max_iter, random=True))

def get_final_coefs(vals_dataframe):
    """Return the last row of a set of coefficients (like those returned by graddescent)."""
    return vals_dataframe[-1:].values[0,:]  # the [0,:] turns a (1,2) into a (2,)


# Corinne's

# Part (a)
def soft_threshold(c, lambduh):
    if c < -lambduh:
        return c+lambduh
    elif c > lambduh:
        return c-lambduh
    else:
        return 0

# Part (b)
def coord_descent_solution(j, X, lambduh, beta, a, xy):
    c_j = 2*(xy[j] - np.sum(X[:,j]*(np.inner(X, beta) - X[:, j]*beta[j])))
    return soft_threshold(c_j, lambduh)/a[j]

def corinne_cycliccoorddescent(beta, X, y, lambduh, max_iter):
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
def corinne_randcoorddescent(beta, X, y, lambduh, max_iter):
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
