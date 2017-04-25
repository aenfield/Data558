import numpy as np
import pandas as pd

def lasso_objective(beta, x, y, lam):
    obj = (1 / len(x)) * sum((y - x.dot(beta))**2)
    obj = obj + (lam * (sum(abs(beta))))
    return obj


def c_term(beta, x, y, j, lam):
    # list of items to keep, with item j removed
    idxs_to_keep = list(range(x.shape[1]))
    idxs_to_keep.remove(j-1)

    beta_minus_j = beta[idxs_to_keep]
    x_minus_j = x[:, idxs_to_keep]

    return 2 * sum(x[:, j-1] * (y - x_minus_j.dot(beta_minus_j)))


def a_term(x, j):
    return 2 * sum(x[:, j-1]**2)


def minimize_beta_term(beta, x, y, j, lam):
    c = c_term(beta, x, y, j, lam)

    if c > lam:
        return (c - lam) / a_term(x, j)
    elif c < -lam:
        return (c + lam) / a_term(x, j)
    else:
        # c is between -lambda and +lambda
        return 0


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


def cycliccoorddescent(x, y, lam, max_iter=500):
    return coorddescent(x, y, lam, get_sequence_of_js(x.shape[1], max_iter))

def randcoorddescent(x, y, lam, max_iter=500):
    return coorddescent(x, y, lam, get_sequence_of_js(x.shape[1], max_iter, random=True))

def get_final_coefs(vals):
    """Return the last row of a set of coefficients (like those returned by graddescent)."""
    return vals[-1:].values[0,:]  # the [0,:] turns a (1,2) into a (2,)