import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import KFold
import seaborn as sns

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

def plot_multiclass_confusion_matrix(cm, classifier_labels):
    ax = sns.heatmap(cm, annot=True, xticklabels=classifier_labels, yticklabels=classifier_labels)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')



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

def get_train_tst_balanced_set(class_label, X_full, y_full, test_prop=0.3, random_state=None):
    """
    Returns an X_train, X_test, y_train, y_test tuple containing a training set that has an 
    equal number of observations with class_label and not class_label. We want this, instead of just
    the standard train_test_split method, so we can ensure that we have train and test sets that
    use all of the class observations (and other observations) without duplicating them. The test
    set has just those with the particular class (since we combine with other classifiers and don't
    know how to classify, individually, non class stuff).
    """
    set_random_state_if_provided(random_state)

    n_each = sum(y_full == class_label)
    train_instances_each = int(np.floor((1 - test_prop) * n_each))

    # get indices of observations with specified class, splitting by by test_prop
    all_class_indices = np.where(y_full == class_label)[0]
    class_train_indices = np.random.choice(all_class_indices, train_instances_each, replace=False)
    class_test_indices = np.setdiff1d(all_class_indices, class_train_indices)

    # and get not this class indices
    notclass_train_indices = np.random.choice(np.where(y_full != class_label)[0], train_instances_each, replace=False)

    train_indices = np.concatenate([class_train_indices, notclass_train_indices])

    return X_full[train_indices], X_full[class_test_indices], y_full[train_indices], y_full[class_test_indices]


def get_classifier_for_label(classifier_label, X, labels, lam, sets=None, random_state=None, max_iters=default_max_iters):
    set_random_state_if_provided(random_state)

    # if we don't get a 4-tuple with the train/test sets, then get our own here
    if sets is None:
        X_train, X_test, labels_train, labels_test = get_train_tst_balanced_set(classifier_label, X, labels)
    else:
        X_train, X_test, labels_train, labels_test = sets

    y_train = np.where(labels_train == classifier_label, 1, -1)

    results_incl_label = fastgradalgo(
        X_train, y_train, t_init=t_init,
        grad_func=compute_gradient_logistic_regression,
        obj_func=compute_objective_logistic_regression,
        lam=lam, max_iter=max_iters, t_func=backtracking)

    return get_final_coefs(results_incl_label).ravel(), X_test, labels_test


def get_results_for_lambdas(classifier_labels, X, labels, lambdas, sets_for_labels=None, random_state=None, max_iters=default_max_iters):
    """
    Given a set of lambda values, one for each classifier label, build classifiers for each label, and
    then predict results in a one-vs-rest fashion, and calculate and return the overall misclassification 
    error. Also return the confusion matrix, as other parts of the project need it; same for the
    predicted labels. With sets_for_labels we also have the ability to accept a iterable with a train-test set (4-tuple)
    for each label in classifier_labels, to enable the caller to pass in known (and unchanging) train-test sets.
    """
    set_random_state_if_provided(random_state)

    # if we don't get a list of 4-tuples with the train/test sets, then we'll generate our own here, to pass in
    if sets_for_labels is None:
        sets_for_labels = [get_train_tst_balanced_set(classifier_label, X, labels) for classifier_label in classifier_labels]

    classifiers_and_test_data = [get_classifier_for_label(classifier_label, X, labels, lam, sets_for_this_label, max_iters=max_iters) for
                                 classifier_label, lam, sets_for_this_label in zip(classifier_labels, lambdas, sets_for_labels)]

    # there's likely a better way to pull out each set of data rather than going through the list multiple times
    # the list's small and this should be quick, I'm guessing, so I won't worry about it for now at least
    classifiers = np.asarray([classifier for classifier, _, _ in classifiers_and_test_data])
    X_test = np.concatenate([test_set for _, test_set, _ in classifiers_and_test_data])
    labels_test = np.concatenate([train_set for _, _, train_set in classifiers_and_test_data])

    predictions_by_classifier = classifiers.dot(X_test.T).T
    label_index_of_highest_prediction = np.argmax(predictions_by_classifier, 1)
    predicted_labels = np.array([classifier_labels[index] for index in label_index_of_highest_prediction])

    misclassification_error = 1 - accuracy_score(labels_test, predicted_labels)
    cm = confusion_matrix(labels_test, predicted_labels)

    return misclassification_error, cm, labels_test, predicted_labels



# ---
# Utils
# ---
def set_random_state_if_provided(random_state):
    if random_state is not None:
        np.random.seed(random_state)