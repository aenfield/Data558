import copy
import numpy as np

def oja_fit(Z, component_count, eta_0, t_0, num_epochs):
    """
    Return an array of arrays where each array is a principal component vector. Uses the Oja algorithm.
    """
    a_0 = np.random.randn(np.size(Z, 1))  # starting point
    a_0 /= np.linalg.norm(a_0, axis=0)
    v1, _ = oja(copy.deepcopy(Z), a_0, eta_0, t_0, num_epochs)  # Run the algorithm for first component vector

    princ_comps = v1[np.newaxis]
    # newaxis because right now we're only returning one and we want it to be element zero in an outer array

    Z_new = copy.deepcopy(Z)
    v_curr = v1
    for i in range(1, component_count):
        Z_new = deflate(Z_new, v_curr)
        v_curr, _ = oja(copy.deepcopy(Z_new), a_0, eta_0, t_0, num_epochs)

        princ_comps = np.concatenate( [princ_comps, v_curr[np.newaxis]] )

    return(princ_comps)

    # TODO do we care about the eigenvalues? If so, perhaps return as an additional element in a tuple an array of the values


def oja(Z, a_0, eta_0, t_0, num_epochs):
    """
    Implements the normalized Oja algorithm to produce the first PCA
    eigenvector (you need to use deflate to get the second and later 
    eigenvectors. I worked through the code below, based on the lab
    and I (think) I understand how it operates.
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

