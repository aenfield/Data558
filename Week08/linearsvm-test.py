import unittest
from linearsvm import *

import numpy as np

# simple data for some of the tests
x_simple = np.array([3,2,0,1,-1,-2]).reshape(3,2)
y_simple = np.array([1,0,0])

class LinearSvmFunctionsTest(unittest.TestCase):
    def test_gradient_with_simpledata(self):
        betas = compute_linearsvm_gradient(np.zeros(2), x_simple, y_simple, 1)
        np.testing.assert_array_almost_equal(betas, np.array([-2, -4/3]))

    def test_gradient_with_nonzero_simpledata(self):
        betas = compute_linearsvm_gradient(np.array([1,1]), x_simple, y_simple, 1)
        np.testing.assert_array_almost_equal(betas, np.array([2, 2]))

    def test_gradient_with_nonzero_simpledata_with_fewer_zeros(self):
        betas = compute_linearsvm_gradient(np.array([1,-1]), x_simple, np.array([3,2,1]), 1)
        np.testing.assert_array_almost_equal(betas, np.array([2, -6]))

    def test_gradient_with_nonzero_simpledata_and_diff_lambda(self):
        betas = compute_linearsvm_gradient(np.array([1,-1]), x_simple, np.array([3,2,1]), 3)
        np.testing.assert_array_almost_equal(betas, np.array([6, -10]))


    def test_objective_with_simpledata(self):
        np.testing.assert_array_almost_equal(compute_linearsvm_objective(np.zeros(2), x_simple, y_simple, 1), 1)

    def test_objective_with_nonzero_simpledata(self):
        np.testing.assert_array_almost_equal(compute_linearsvm_objective(np.array([1,1]), x_simple, y_simple, 1), 2.667, decimal=3)

    def test_objective_with_nonzero_simpledata_and_diff_lambda(self):
        np.testing.assert_array_almost_equal(compute_linearsvm_objective(np.array([1,1]), x_simple, y_simple, 3), 6.667, decimal=3)


class GradientImplementationTest(unittest.TestCase):
    pass


if __name__ == '__main__':
    unittest.main()