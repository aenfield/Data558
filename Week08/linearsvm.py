import numpy as np


def computelinearsvmgrad(beta, x, y, lam):
    return np.zeros(2)
    # grad_beta = (-2 / len(x) * (x.T).dot((y - x.dot(beta))))
    # grad_beta = grad_beta + 2 * lam * beta
    # return grad_beta


def computeobject():
    pass


def graddescent():
    pass


def fastgradalgo():
    pass
