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


def cycliccoorddescent(x, y, lam, max_iter=500):
    feature_count = x.shape[1]

    # get a sequence of feature indices/j values
    # we repeat enough times to have at least or more numbers,
    # then filter to just the right amount
    repeat_features_count = (max_iter / feature_count) + 1
    sequence_of_js = np.repeat(y[np.newaxis], 2, 0).flatten()[:max_iter]

    for j in sequence_of_js:
