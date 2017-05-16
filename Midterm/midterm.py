import numpy as np
import pandas as pd


# this is the equivalent of my computeobj function
def elasticnet_objective(beta, x, y, lam, alpha):
    obj = (1 / len(x)) * np.sum((y - x.dot(beta))**2)
    obj = obj + (lam * alpha * (np.sum(np.abs(beta))))
    obj = obj + (lam * (1-alpha) * np.sum(np.square(beta)))
    return obj

def c_term(beta, x, y, j):
    # list of items to keep, with item j removed
    idxs_to_keep = list(range(x.shape[1]))
    idxs_to_keep.remove(j-1)

    beta_minus_j = beta[idxs_to_keep]
    x_minus_j = x[:, idxs_to_keep]

    return (2 / len(x)) * np.sum(x[:, j-1] * (y - x_minus_j.dot(beta_minus_j)))

def a_term(x, j, lam, alpha):
    return 2 * ( ((1/len(x)) * np.sum(x[:, j-1]**2)) + lam - (lam * alpha) )

# computes the formula of the solution of the elasticnet minimization problem
def minimize_beta_term(beta, x, y, j, lam, alpha):
    c = c_term(beta, x, y, j)
    lamalpha = lam * alpha

    if c > lamalpha:
        return (c - lamalpha) / a_term(x, j, lam, alpha)
    elif c < -lamalpha:
        return (c + lamalpha) / a_term(x, j, lam, alpha)
    else:
        # c is between -lambda and +lambda
        return 0

def coorddescent(x, y, lam, alpha, j_sequence):
    """
    Implements coordinate descent, using the provided sequence, which defines
    whether the params are handled cyclically or randomly
    """
    feature_count = x.shape[1]
    beta = np.zeros(feature_count)
    saved_beta_vals = pd.DataFrame(beta[np.newaxis, :])

    for j in j_sequence:
        b_hat_for_j = minimize_beta_term(beta, x, y, j, lam, alpha)
        beta_new_val = beta.copy()  # modify copy so we don't retroactively modify rows because of by reference
        beta_new_val[j-1] = b_hat_for_j # j indices are 1-based, so subtract one to set correct coef
        saved_beta_vals = saved_beta_vals.append(pd.DataFrame(beta_new_val[np.newaxis]), ignore_index=True)
        beta = beta_new_val

    return saved_beta_vals

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


def cycliccoorddescent(x, y, lam, alpha, max_iter=500):
    return coorddescent(x, y, lam, alpha, get_sequence_of_js(x.shape[1], max_iter))

def randcoorddescent(x, y, lam, alpha, max_iter=500):
    return coorddescent(x, y, lam, alpha, get_sequence_of_js(x.shape[1], max_iter, random=True))

def get_final_coefs(vals_dataframe):
    """Return the last row of a set of coefficients (like those returned by graddescent)."""
    return vals_dataframe[-1:].values[0,:]  # the [0,:] turns a (1,2) into a (2,)