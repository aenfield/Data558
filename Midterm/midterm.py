import numpy as np
import pandas as pd
import copy

# ----
# Elastic net
# ----

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


# ----
# PCA via Oja
# ----


def oja_fit(Z, component_count, eta_0, t_0, num_epochs):
    """
    Return an array of arrays where each array is a principal component vector. Uses the Oja algorithm. I
    put this convenience routine together based on Corinne's lab code.
    """
    a_0 = np.random.randn(np.size(Z, 1))  # starting point
    a_0 /= np.linalg.norm(a_0, axis=0)
    v1, lambdas = oja(copy.deepcopy(Z), a_0, eta_0, t_0, num_epochs)  # Run the algorithm for first component vector

    # newaxis because right now we're only returning one and we want it to be element zero in an outer array
    princ_comps = v1[np.newaxis]
    eigenvalues = lambdas[-1][np.newaxis]
    print(eigenvalues)

    Z_new = copy.deepcopy(Z)
    v_curr = v1
    for i in range(1, component_count):
        Z_new = deflate(Z_new, v_curr)
        v_curr, lambdas_curr = oja(copy.deepcopy(Z_new), a_0, eta_0, t_0, num_epochs)
        print(lambdas_curr[-1])

        princ_comps = np.concatenate( [princ_comps, v_curr[np.newaxis]] )
        eigenvalues = np.concatenate( [eigenvalues, lambdas_curr[-1][np.newaxis]] )

    return(princ_comps, eigenvalues)


def oja(Z, a_0, eta_0, t_0, num_epochs):
    """
    Implements the normalized Oja algorithm to produce the first PCA eigenvector (and then we use deflate to 
    get the second and later eigenvectors). I worked through the code from Corinne below, based on the lab,
    and understand how it operates.
    """
    t = 0
    a = a_0
    n = np.size(Z, 0)
    lambdas = np.zeros(num_epochs)
    for epoch in range(0, num_epochs):
        # Shuffle the rows of the data after each epoch
        np.random.shuffle(Z)
        for i in range(0, n):
            # Note it's faster not to compute the matrix ZZ^T
            last_a = a
            a = a + eta_0/(t+t_0)*np.dot(Z[i,], np.dot(Z[i,].T, a))
            a = a/np.linalg.norm(a)

            a_delta = last_a - a
            mean_a_delta = np.mean(a_delta)

            t += 1
        lambdas[epoch] = a.dot(Z.T).dot(Z).dot(a)/n

    return a, lambdas

def deflate(Z, a):
    return Z - Z.dot(np.outer(a, a))
