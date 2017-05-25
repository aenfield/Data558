import numpy as np
import pandas as pd


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

# TODO check this against Corinne's implementation
def backtracking(coefs, x, y, grad_func, obj_func, t=1,
                 alpha=0.5, beta=0.5, max_iter=100, lam=5):
    """
    Returns a value of t, for use w/ gradient descent, obtained via
    backtracking."""
    grad_coefs = grad_func(coefs, x, y, lam)
    norm_grad_coefs = np.linalg.norm(grad_coefs) # norm of gradient
    found_t = False
    iter = 0
    while (found_t == False and iter < max_iter):
        attempted_value = obj_func(coefs - (t * grad_coefs),
                                   x, y, lam)
        compare_value = obj_func(coefs, x, y,
                                 lam) - (alpha * t * norm_grad_coefs**2)
        if attempted_value < compare_value:
          found_t = True
        elif iter == max_iter:
            # should ideally check whether 'exit' is valid
            exit("Backtracking: max iterations reached")
        else:
            t = t * beta
            iter = iter + 1

    return t
    #return(t, iter)

# TODO check this against Corinne's impl
def fastgradalgo(x, y, t_init, grad_func, obj_func,
                 max_iter=100, lam=5, t_func=None):
    """
    Implement the fast gradient descent algorithm. Uses t_func 
    to determine t, or uses a fixed size of t_init if
    t_func is None. That is, pass in a function like 
    'backtracking' to do a line search.
    """
    beta = np.zeros(x.shape[1])  # shape[1] is the number of columns
    theta = beta.copy()  # we init theta to zeros too
    t = t_init
    beta_vals = pd.DataFrame(beta[np.newaxis, :])

    for i in range(0, max_iter):
        if t_func:
            t = t_func(beta, x, y, grad_func, obj_func, t_init)

        beta_prev = beta.copy()
        beta = theta - (t * grad_func(theta, x, y, lam))
        theta = beta + (i / (i + 3)) * (beta - beta_prev)

        beta_vals = beta_vals.append(pd.DataFrame(beta[np.newaxis]),
                                     ignore_index=True)

    return beta_vals

def get_final_coefs(beta_vals):
    return beta_vals[-1:].values