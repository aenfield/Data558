import unittest
from finalproj import *

import numpy as np
from sklearn import preprocessing
import pickle


# simple data for some of the tests
x_simple = np.array([3,2,0,1,-1,-2]).reshape(3,2)
y_simple = np.array([1,0,0])
beta_zeros = np.zeros(2)
beta_ones = np.array([1,1])
beta_onezero = np.array([1,-1])
lam = 1

# we'll go ahead and load the actual data too, for some tests
features_train = np.array(pickle.load(open('features_train', 'rb')))
labels_train = np.array(pickle.load(open('labels_train', 'rb')))
X_scaled = preprocessing.scale(features_train)


# Corinne's implementations, to get the correct results to use to test my implementation
# (keep these here as opposed to in the corinne.py file, as the production code doesn't depend on either,
#  and the tests only depend on these)
def corinne_computegrad_logistic(beta, x, y, lambduh):
    yx = y[:, np.newaxis]*x
    denom = 1 + np.exp(-yx.dot(beta))
    grad = 1/len(y)*np.sum(-yx*np.exp(-yx.dot(beta[:, np.newaxis])) / denom[:, np.newaxis], axis=0) + 2*lambduh*beta
    return grad

def corinne_objective_logistic(beta, x, y, lambduh):
    return 1/len(y) * np.sum(np.log(1 + np.exp(-y*x.dot(beta)))) + lambduh * np.linalg.norm(beta)**2


class LinearSvmFunctionsTest(unittest.TestCase):
    def test_gradient_with_simpledata(self):
        betas = compute_gradient_logistic_regression(beta_zeros, x_simple, y_simple, lam)
        np.testing.assert_array_almost_equal(betas, corinne_computegrad_logistic(beta_zeros, x_simple, y_simple, lam))

    def test_gradient_with_nonzero_simpledata(self):
        betas = compute_gradient_logistic_regression(beta_ones, x_simple, y_simple, lam)
        np.testing.assert_array_almost_equal(betas, corinne_computegrad_logistic(beta_ones, x_simple, y_simple, lam))

    def test_gradient_with_nonzero_simpledata_with_fewer_zeros(self):
        betas = compute_gradient_logistic_regression(beta_onezero, x_simple, np.array([3,2,1]), lam)
        np.testing.assert_array_almost_equal(betas, corinne_computegrad_logistic(beta_onezero, x_simple, np.array([3, 2, 1]), lam))

    def test_gradient_with_nonzero_simpledata_and_diff_lambda(self):
        betas = compute_gradient_logistic_regression(beta_onezero, x_simple, np.array([3,2,1]), 3)
        np.testing.assert_array_almost_equal(betas, corinne_computegrad_logistic(beta_onezero, x_simple, np.array([3, 2, 1]), 3))


    def test_objective_with_simpledata(self):
        val = compute_objective_logistic_regression(beta_zeros, x_simple, y_simple, lam)
        np.testing.assert_array_almost_equal(val, corinne_objective_logistic(beta_zeros, x_simple, y_simple, lam), decimal=3)

    def test_objective_with_nonzero_simpledata(self):
        val = compute_objective_logistic_regression(beta_ones, x_simple, y_simple, lam)
        np.testing.assert_array_almost_equal(val, corinne_objective_logistic(beta_ones, x_simple, y_simple, lam), decimal=3)

    def test_objective_with_nonzero_simpledata_and_diff_lambda(self):
        val = compute_objective_logistic_regression(beta_ones, x_simple, y_simple, 3)
        np.testing.assert_array_almost_equal(val, corinne_objective_logistic(beta_ones, x_simple, y_simple, 3), decimal=3)


class OneVsRestTest(unittest.TestCase):

    def test_get_balanced_observations(self):
        class_label = '086.Pacific_Loon'
        X, y = get_balanced_set(class_label, X_scaled, labels_train)
        n_class = len(np.where(y == class_label))
        n_notclass = len(np.where(y != class_label))
        self.assertEqual(n_class, n_notclass)



# TODO Would be nice to have tests for the cross-validation stuff
# TODO Would be nice to have tests for the accuracy calculation stuff


if __name__ == '__main__':
    unittest.main()