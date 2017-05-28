import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import KFold

t_init = 0.01
default_max_iters = 300


# ---
# Gradient and objective functions
# ---

def compute_gradient_logistic_regression(beta, x, y, lam=5):
    exp_term = np.exp(-y * (x.dot(beta)))
    scalar_term = (y * exp_term) / (1 + exp_term)
    penalty = 2*lam*beta
    grad_beta = (1/len(x)) * -scalar_term[np.newaxis].dot(x) + penalty
    return grad_beta.flatten() # flatten to return a vector instead of a 1-D array

def compute_objective_logistic_regression(beta, x, y, lam=5):
    obj = 1/len(x) * sum( np.log(1 + np.exp(-y * (x.dot(beta)))) )
    obj = obj + lam*(np.linalg.norm(beta))**2
    return obj





# ---
# Fast gradient descent
# ---

# my implementation from homework three
def fastgradalgo(x, y, t_init, grad_func, obj_func, max_iter=100, lam=5, t_func=None):
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

        beta_vals = beta_vals.append(pd.DataFrame(beta[np.newaxis]), ignore_index=True)

    return beta_vals

# my implementation from homework three
def get_final_coefs(beta_vals):
    return beta_vals[-1:].values

# this is my implementation from homework three
def backtracking(coefs, x, y, grad_func, obj_func, t=1, alpha=0.5, beta=0.5, max_iter=100, lam=5):
    """
    Returns a value of t, for use w/ gradient descent, obtained via backtracking."""
    grad_coefs = grad_func(coefs, x, y, lam)
    norm_grad_coefs = np.linalg.norm(grad_coefs) # norm of gradient
    found_t = False
    iter = 0
    while (found_t == False and iter < max_iter):
        attempted_value = obj_func(coefs - (t * grad_coefs), x, y, lam)
        compare_value = obj_func(coefs, x, y, lam) - (alpha * t * norm_grad_coefs**2)
        if attempted_value < compare_value:
          found_t = True
        elif iter == max_iter:
            # TODO should ideally check whether 'exit' is valid
            exit("Backtracking: max iterations reached")
        else:
            t = t * beta
            iter = iter + 1

    return t
    #return(t, iter)


# ---
# Plotting
# ---

def get_misclassification_errors_by_iteration(beta_results_df, X, y):
    return beta_results_df.apply(lambda r: 1 - get_accuracy(r.values, X, y), axis=1)

def plot_misclassification_errors_by_iteration(results_df, X_train, X_test, y_train, y_test):
    errors = pd.DataFrame({'Train': get_misclassification_errors_by_iteration(results_df, X_train, y_train),
                           'Validation': get_misclassification_errors_by_iteration(results_df, X_test, y_test)})
    ax = errors.plot()
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Misclassification error')
    ax.set_title('Misclassification error by iteration')

# ---
# Metrics
# ---

def get_probability(logodds):
    return 1 / (1 + np.exp(-logodds))

def get_y_pred(beta_coefs, X, y, prob_threshold=0.5):
    y_pred = X.dot(beta_coefs.T).ravel()  # ravel to convert to vector

    # for logistic regression convert to a prob and use a prob threshold
    probs = get_probability(y_pred)
    y_thresholded = np.where(probs > prob_threshold, 1, -1)
    return y_thresholded

def get_accuracy(beta_coefs, X, y_actual, prob_threshold=0.5):
    """
    Return the classification accuracy given a set of coefficients, in 
    beta_coefs, and observations, in X, compared to actual/known values 
    in y_actual. The threshold parameter defines the value above which the
    predicted value is considered a positive example.
    """
    return accuracy_score(y_actual, get_y_pred(beta_coefs, X, y_actual, prob_threshold))

def get_confusion_matrix(beta_coefs, X, y_actual, prob_threshold=0.5):
    """
    Return the confusion matrix given a set of coefficients, in 
    beta_coefs, and observations, in X, compared to actual/known values 
    in y_actual. The threshold parameter defines the value above which the
    predicted value is considered a positive example. In this confusion
    matrix, the y-axis is actual and the x-axis is predicted, so it looks like
    the following (the labels param in the call makes this possible):
    -----------
    | TP | FN |
    -----------
    | FP | TN | 
    -----------
    (Note that switching the labels like this makes the sns.heatmap function 
    assign incorrect tick marks.)
    """
    return confusion_matrix(y_actual, get_y_pred(beta_coefs, X, y_actual, prob_threshold), labels=[1,-1])

# ---
# Cross validation
# ---

# Since we need to do this ourselves, per the assignments.

# using 'tst' so this function isn't incorrectly detected as a unit test (there's probably a way to fix this w/o changing the function's name)
def train_and_tst_single_fold(X_full, y_full, lam, train_index, test_index, max_iters=default_max_iters):
    """
    Train using the data identified by the indices in train_index, and then test
    (and return accuracy) using the data identified by the indices in test_index.
    """
    beta_vals = fastgradalgo(
        X_full[train_index], y_full[train_index], t_init=t_init,
        grad_func=compute_gradient_logistic_regression,
        obj_func=compute_objective_logistic_regression,
        lam=lam, max_iter=max_iters)

    final_coefs = get_final_coefs(beta_vals)

    return get_accuracy(final_coefs, X_full[test_index], y_full[test_index])

# using 'tst' so this function isn't incorrectly detected as a unit test (there's probably a way to fix this w/o changing the function's name)
def train_and_tst_for_all_folds(num_folds, X_full, y_full, train_indices, test_indices, lam, max_iters=default_max_iters):
    """
    Train and test for all folds. Return the mean of the set of accuracy scores from all folds.
    """
    accuracy_scores = [train_and_tst_single_fold(X_full, y_full, lam, train_indices[i], test_indices[i], max_iters) for i in range(num_folds)]
    return(np.mean(accuracy_scores))

def cross_validate(num_folds, X, y, lambdas, max_iters=default_max_iters, random_state=42):
    # get arrays with num_folds sets of test and train indices - one for each fold
    kf = KFold(num_folds, shuffle=True, random_state=random_state)

    train_indices_list = []
    test_indices_list = []
    for train_index, test_index in kf.split(X):
        train_indices_list.append(train_index)
        test_indices_list.append(test_index)

    train_indices = np.array(train_indices_list)
    test_indices = np.array(test_indices_list)

    # do num folds-fold cross validation for each value of lambda, and
    # save the mean of each set's classification accuracy
    accuracy_values_by_lambda = [train_and_tst_for_all_folds(num_folds, X, y, train_indices, test_indices, lam, max_iters) for lam in lambdas]

    return lambdas, accuracy_values_by_lambda


# ---
# One vs. rest
# ---

def get_balanced_set(class_label, X_full, y_full, random_state=None):
    """
    Returns an X, y tuple containing a training set that has an equal number of observations with
    class_label and not class_label. Done as simple as possible, without generalizations that I'd
    normally include (I won't gold plate it since I know the data with which it's being used).
    """
    if random_state is not None:
        np.random.seed(random_state)

    n_each = sum(y_full == class_label)
    notclass_indices = np.random.choice(np.where(y_full != class_label)[0], n_each, replace=False)

    all_indices = np.concatenate([np.where(y_full == class_label)[0],
                                  np.random.choice(np.where(y_full != class_label)[0], n_each, replace=False)])

    return X_full[all_indices], y_full[all_indices]