import copy
import numpy as np


def oja(Z, a_0, eta_0, t_0, num_epochs):
    """
    Implements the normalized Oja algorithm to produce the first PCA
    eigenvector. Use deflate to get the second and later eigenvectors.
    Note that I'm using Corinne's code - I've worked through it and
    (think) I understand how it operates.
    :param Z: Centered data matrix
    :param a_0: Initial starting point
    :param eta_0: Parameter in the numerator of the step size
    :param t_0: Parameter in the denominator of the step size
    :param num_epochs: Number of passes through the data before we stop
    :return: a: a tuple with a, the estimated eigenvector corresponding
     to the maximum eigenvalue of 1/N*Z^TZ, and an array of eigenvalues
     corresponding to each epoch considered during processing (so the
     last value is the estimated top eigenvalue.
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
    """
    TODO add docs and params
    :param Z: 
    :param a: 
    :return: 
    """
    return Z - Z.dot(np.outer(a, a))

def oja_fit(Z, component_count, eta_0, t_0, num_epochs):
    """
    TODO add docs and params
    :param Z: 
    :param component_count: s
    :return: 
    """
    a_0 = np.random.randn(np.size(Z, 1))  # starting point
    a_0 /= np.linalg.norm(a_0, axis=0)
    v1, _ = oja(copy.deepcopy(Z), a_0, eta_0, t_0, num_epochs)  # Run the algorithm for first component vector

    princ_comps = v1[np.newaxis] # newaxis because right now we're only returning one and we want it to be element zero in an outer array

    Z_new = copy.deepcopy(Z)
    v_curr = v1
    for i in range(1, component_count):
        Z_new = deflate(Z_new, v_curr)
        v_curr, _ = oja(copy.deepcopy(Z_new), a_0, eta_0, t_0, num_epochs)

        princ_comps = np.concatenate( [princ_comps, v_curr[np.newaxis]] )
    #
    # Z1 = deflate(Z, v1)
    # v2, lambdas2 = oja(copy.deepcopy(Z1), a_0, .00001, 1, 50)

    return(princ_comps)

    # TODO do we care about/need to return the eigenvalues? If so, perhaps return as an additional element in a tuple an array of the values
