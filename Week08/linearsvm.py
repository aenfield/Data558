import numpy as np


def compute_linearsvm_gradient(beta, x, y, lam):
    grad_beta_max_term = np.maximum(0, 1 - (y * (x.dot(np.array(beta)))))[np.newaxis].T
    grad_beta_sum_term = y[np.newaxis].T * x * grad_beta_max_term
    grad_beta_without_penalty = (-2 / len(x)) * np.sum(grad_beta_sum_term, 0)
    grad_beta_penalty = 2 * lam * beta
    return grad_beta_without_penalty + grad_beta_penalty

def compute_linearsvm_objective(beta, x, y, lam):
    objective_max_term = np.maximum(0, 1 - (y * (x.dot(np.array(beta)))))[np.newaxis].T
    obj_without_penalty = (1 / len(x)) * np.sum(objective_max_term**2)
    obj_penalty = lam * np.sum(beta**2)

    return obj_without_penalty + obj_penalty

def fastgradalgo():
    pass
