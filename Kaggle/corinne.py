#from __future__ import division
import matplotlib.pyplot as plt
import numpy as np
# import pandas as pd
# import scipy.linalg
# import sklearn.linear_model
# import sklearn.preprocessing

# Part (c): Read in the data, standardize it
# spam = pd.read_table('https://statweb.stanford.edu/~tibs/ElemStatLearn/datasets/spam.data', sep=' ', header=None)
# test_indicator = pd.read_table('https://statweb.stanford.edu/~tibs/ElemStatLearn/datasets/spam.traintest', sep=' ',
#                                header=None)
#
# x = np.asarray(spam)[:, 0:-1]
# y = np.asarray(spam)[:, -1]*2 - 1  # Convert to +/- 1
# test_indicator = np.array(test_indicator).T[0]
#
# # Divide the data into train, test sets
# x_train = x[test_indicator == 0, :]
# x_test = x[test_indicator == 1, :]
# y_train = y[test_indicator == 0]
# y_test = y[test_indicator == 1]
#
# # Standardize the data.
# scaler = sklearn.preprocessing.StandardScaler()
# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)
#
# # Keep track of the number of samples and dimension of each sample
# n_train = len(y_train)
# n_test = len(y_test)
# d = np.size(x, 1)


# Part (d): Function for the gradient
#def computegrad(beta, lambduh, x=x_train, y=y_train):
def computegrad(beta, lambduh, x, y):
    yx = y[:, np.newaxis]*x
    denom = 1+np.exp(-yx.dot(beta))
    grad = 1/len(y)*np.sum(-yx*np.exp(-yx.dot(beta[:, np.newaxis]))/denom[:, np.newaxis], axis=0) + 2*lambduh*beta
    return grad


# Part (e): Backtracking line search (and objective function)
#def objective(beta, lambduh, x=x_train, y=y_train):
def objective(beta, lambduh, x, y):
    return 1/len(y) * np.sum(np.log(1 + np.exp(-y*x.dot(beta)))) + lambduh * np.linalg.norm(beta)**2

#def bt_line_search(beta, lambduh, eta=1, alpha=0.5, betaparam=0.8, maxiter=100, x=x_train, y=y_train):
def bt_line_search(beta, x, y, lambduh, eta=1, alpha=0.5, betaparam=0.8, maxiter=100):
    grad_beta = computegrad(beta, lambduh, x=x, y=y)
    norm_grad_beta = np.linalg.norm(grad_beta)
    found_eta = 0
    iter = 0
    while found_eta == 0 and iter < maxiter:
        if objective(beta - eta * grad_beta, lambduh, x=x, y=y) < objective(beta, lambduh, x=x, y=y) \
                - alpha * eta * norm_grad_beta ** 2:
            found_eta = 1
        elif iter == maxiter - 1:
            print('Warning: Max number of iterations of backtracking line search reached')
        else:
            eta *= betaparam
            iter += 1
    return eta


# Part (f): Gradient descent algorithm
#def graddescent(beta_init, lambduh, eta_init, maxiter, x=x_train, y=y_train):
def graddescent(beta_init, lambduh, eta_init, maxiter, x, y):
    beta = beta_init
    grad_beta = computegrad(beta, lambduh, x=x, y=y)
    beta_vals = beta
    iter = 0
    while iter < maxiter:
        eta = bt_line_search(beta, x=x, y=y, lambduh=lambduh, eta=eta_init)
        beta = beta - eta*grad_beta
        # Store all of the places we step to
        beta_vals = np.vstack((beta_vals, beta))
        grad_beta = computegrad(beta, lambduh, x=x, y=y)
        iter += 1
        if iter % 100 == 0:
            print('Gradient descent iteration', iter)
    return beta_vals


# Part (g): gradient algorithm
#def fastgradalgo(beta_init, theta_init, lambduh, eta_init, maxiter, x=x_train, y=y_train):
def fastgradalgo(beta_init, theta_init, lambduh, eta_init, maxiter, x, y):
    beta = beta_init
    theta = theta_init
    grad_theta = computegrad(theta, lambduh, x=x, y=y)
    beta_vals = beta
    theta_vals = theta
    iter = 0
    while iter < maxiter:
        eta = bt_line_search(theta, x=x, y=y, lambduh=lambduh, eta=eta_init)
        beta_new = theta - eta*grad_theta
        theta = beta_new + iter/(iter+3)*(beta_new-beta)
        # Store all of the places we step to
        beta_vals = np.vstack((beta_vals, beta_new))
        theta_vals = np.vstack((theta_vals, theta))
        grad_theta = computegrad(theta, lambduh, x=x, y=y)
        beta = beta_new
        iter += 1
        if iter % 100 == 0:
            print('Fast gradient iteration', iter)
    return beta_vals, theta_vals


# Part (h)
#def objective_plot(betas_gd, betas_fg, lambduh, x=x_train, y=y_train, save_file=''):
def objective_plot(betas_1, lambduh, x, y, save_file='', betas_2=None):
    num_points = np.size(betas_1, 0)
    objs_1 = np.zeros(num_points)
    #objs_2 = np.zeros(num_points)
    for i in range(0, num_points):
        objs_1[i] = objective(betas_1[i, :], lambduh, x=x, y=y)
        #objs_2[i] = objective(betas_2[i, :], lambduh, x=x, y=y)
    fig, ax = plt.subplots()
    ax.plot(range(1, num_points + 1), objs_1, label='betas_1')
    #ax.plot(range(1, num_points + 1), objs_2, c='red', label='fast gradient')
    plt.xlabel('Iteration')
    plt.ylabel('Objective value')
    plt.title('Objective value vs. iteration when lambda='+str(lambduh))
    ax.legend(loc='upper right')
    if not save_file:
        plt.show()
    else:
        plt.savefig(save_file)


# lambduh = 0.1
# beta_init = np.zeros(d)
# theta_init = np.zeros(d)
# # See slide 28 in the lecture 3 slides for how to initialize the step size
# eta_init = 1/(scipy.linalg.eigh(1/len(y_train)*x_train.T.dot(x_train), eigvals=(d-1, d-1), eigvals_only=True)[0]+lambduh)
# maxiter = 300
# betas_grad = graddescent(beta_init, lambduh, eta_init, maxiter)
# betas_fastgrad, thetas_fastgrad = fastgradalgo(beta_init, theta_init, lambduh, eta_init, maxiter)
# objective_plot(betas_grad, betas_fastgrad, lambduh, save_file='hw3_q1_part_h_output.png')


# Part (i): Compare to scikit-learn
# lr = sklearn.linear_model.LogisticRegression(penalty='l2', C=1/(2*lambduh*n_train), fit_intercept=False, tol=10e-8, max_iter=1000)
# lr.fit(x_train, y_train)
# print(lr.coef_)
# print(betas_fastgrad[-1, :])
#
# print(objective(betas_fastgrad[-1, :], lambduh))
# print(objective(lr.coef_.flatten(), lambduh))


# Part (j): Run cross-validation to find lambda. Plot the objective values and misclassification errors
def compute_misclassification_error(beta_opt, x, y):
    y_pred = 1/(1+np.exp(-x.dot(beta_opt))) > 0.5
    y_pred = y_pred*2 - 1  # Convert to +/- 1
    return np.mean(y_pred != y)


def plot_misclassification_error(betas_grad, betas_fastgrad, x, y, save_file='', title=''):
    niter = np.size(betas_grad, 0)
    error_grad = np.zeros(niter)
    error_fastgrad = np.zeros(niter)
    for i in range(niter):
        error_grad[i] = compute_misclassification_error(betas_grad[i, :], x, y)
        error_fastgrad[i] = compute_misclassification_error(betas_fastgrad[i, :], x, y)
    fig, ax = plt.subplots()
    ax.plot(range(1, niter + 1), error_grad, label='gradient descent')
    ax.plot(range(1, niter + 1), error_fastgrad, c='red', label='fast gradient')
    plt.xlabel('Iteration')
    plt.ylabel('Misclassification error')
    if title:
        plt.title(title)
    ax.legend(loc='upper right')
    if not save_file:
        plt.show()
    else:
        plt.savefig(save_file)

# lr_cv = sklearn.linear_model.LogisticRegressionCV(penalty='l2', fit_intercept=False, tol=10e-8, max_iter=1000)
# lr_cv.fit(x_train, y_train)
# optimal_lambda = lr_cv.C_[0]
# print('Optimal lambda=', optimal_lambda)
#
# betas_grad = graddescent(beta_init, optimal_lambda, eta_init, maxiter)
# betas_fastgrad, thetas_fastgrad = fastgradalgo(beta_init, theta_init, optimal_lambda, eta_init, maxiter)
#
# objective_plot(betas_grad, betas_fastgrad, optimal_lambda, save_file='hw3_q1_part_j_output1.png')
# plot_misclassification_error(betas_grad, betas_fastgrad, x_train, y_train, save_file='hw3_q1_part_j_output2.png',
#                              title='Training set misclassification error')
# plot_misclassification_error(betas_grad, betas_fastgrad, x_test, y_test, save_file='hw3_q1_part_j_output3.png',
#                              title='Test set misclassification error')








