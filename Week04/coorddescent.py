import numpy as np

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


def get_sequence_of_js(feature_count, iterations):
    # get a sequence of feature indices/j values of size max_iter
    # we repeat enough times to have at least the right size or more,
    # then filter to just the right amount
    repeat_features_count = (iterations // feature_count) + 1
    sequence_of_js = np.repeat(np.arange(1, feature_count+1)[np.newaxis], repeat_features_count, 0).flatten()[:iterations]

    return sequence_of_js


def cycliccoorddescent(x, y, lam, max_iter=500):
    feature_count = x.shape[1]

    beta = np.zeros(feature_count)
    sequence_of_js = get_sequence_of_js(feature_count, max_iter)

    for j in sequence_of_js:
        b_hat_for_j = minimize_beta_term(beta, x, y, j, lam)
        beta[j-1] = b_hat_for_j # j indices are 1-based, so subtract one to set correct coef

    return beta
