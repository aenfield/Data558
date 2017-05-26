import unittest
from finalproj import *

import numpy as np

# TODO would be nice to have at least a few tests for the logistic regression gradient and objective functions

# simple data for some of the tests
x_simple = np.array([3,2,0,1,-1,-2]).reshape(3,2)
y_simple = np.array([1,0,0])

# Corinne's implementations, to get the correct results to use to test my implementation
def corinne_computegrad(beta, x, y, lambduh):
    yx = y[:, np.newaxis]*x
    denom = 1 + np.exp(-yx.dot(beta))
    grad = 1/len(y)*np.sum(-yx*np.exp(-yx.dot(beta[:, np.newaxis])) / denom[:, np.newaxis], axis=0) + 2*lambduh*beta
    return grad

def corinne_objective(beta, x, y, lambduh):
    return 1/len(y) * np.sum(np.log(1 + np.exp(-y*x.dot(beta)))) + lambduh * np.linalg.norm(beta)**2

class LinearSvmFunctionsTest(unittest.TestCase):
    def test_gradient_with_simpledata(self):
        beta_init = np.zeros(2)
        lam = 1
        betas = compute_gradient_logistic_regression(beta_init, x_simple, y_simple, lam)
        np.testing.assert_array_almost_equal(betas, corinne_computegrad(beta_init, x_simple, y_simple, lam))

    # def test_gradient_with_nonzero_simpledata(self):
    #     betas = compute_linearsvm_gradient(np.array([1,1]), x_simple, y_simple, 1)
    #     np.testing.assert_array_almost_equal(betas, np.array([2, 2]))
    #
    # def test_gradient_with_nonzero_simpledata_with_fewer_zeros(self):
    #     betas = compute_linearsvm_gradient(np.array([1,-1]), x_simple, np.array([3,2,1]), 1)
    #     np.testing.assert_array_almost_equal(betas, np.array([2, -6]))
    #
    # def test_gradient_with_nonzero_simpledata_and_diff_lambda(self):
    #     betas = compute_linearsvm_gradient(np.array([1,-1]), x_simple, np.array([3,2,1]), 3)
    #     np.testing.assert_array_almost_equal(betas, np.array([6, -10]))
    #
    #
    # def test_objective_with_simpledata(self):
    #     np.testing.assert_array_almost_equal(compute_linearsvm_objective(np.zeros(2), x_simple, y_simple, 1), 1)
    #
    # def test_objective_with_nonzero_simpledata(self):
    #     np.testing.assert_array_almost_equal(compute_linearsvm_objective(np.array([1,1]), x_simple, y_simple, 1), 2.667, decimal=3)
    #
    # def test_objective_with_nonzero_simpledata_and_diff_lambda(self):
    #     np.testing.assert_array_almost_equal(compute_linearsvm_objective(np.array([1,1]), x_simple, y_simple, 3), 6.667, decimal=3)


# class GradientImplementationTest(unittest.TestCase):
#     pass


if __name__ == '__main__':
    unittest.main()