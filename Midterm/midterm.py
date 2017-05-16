import numpy as np

# this is the equivalent of my computeobj function
def elasticnet_objective(beta, x, y, lam, alpha):
    obj = (1 / len(x)) * np.sum((y - x.dot(beta))**2)
    obj = obj + (lam * alpha * (np.sum(np.abs(beta))))
    obj = obj + (lam * (1-alpha) * np.sum(np.square(beta)))
    return obj
