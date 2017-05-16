import numpy as np

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
