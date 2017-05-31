import unittest
from kernelsvm import *

import numpy as np
from sklearn.metrics.pairwise import linear_kernel, polynomial_kernel

# simple data for some of the tests
x_simple = np.array([3,2,0,1,-1,-2]).reshape(3,2)
y_simple = np.array([1,1,-1])

class KernelSvmFunctionsTest(unittest.TestCase):
    def test_computegram_linear_with_simpledata(self):
        gram = computegram_linear(x_simple)
        np.testing.assert_array_almost_equal(gram, linear_kernel(x_simple))

    def test_computegram_polynomial_with_simpledata(self):
        gram = computegram_polynomial(x_simple, 3, 1)
        np.testing.assert_array_almost_equal(gram, polynomial_kernel(x_simple, degree=3, coef0=1, gamma=1))
        # we use gamma = 1 to match what we want, overriding the function's default that normalizes based on n

    def test_kerneleval_linear_with_simpledata(self):
        vec = kerneleval_linear(x_simple, np.array([2,2]))
        np.testing.assert_array_almost_equal(vec, np.array([10,2,-6]))

    # TODO need to write/test kerneleval_polynomial?

    def test_gradient_with_simpledata(self):
        betas = compute_kernelsvm_gradient(np.zeros(2), x_simple, y_simple, 1)
        np.testing.assert_array_almost_equal(betas, np.array([-2, -4/3]))

    def test_gradient_with_nonzero_simpledata(self):
        betas = compute_kernelsvm_gradient(np.array([1,1]), x_simple, y_simple, 1)
        np.testing.assert_array_almost_equal(betas, np.array([2, 2]))

    def test_gradient_with_nonzero_simpledata_with_fewer_zeros(self):
        betas = compute_kernelsvm_gradient(np.array([1,-1]), x_simple, np.array([3,2,1]), 1)
        np.testing.assert_array_almost_equal(betas, np.array([2, -6]))

    def test_gradient_with_nonzero_simpledata_and_diff_lambda(self):
        betas = compute_kernelsvm_gradient(np.array([1,-1]), x_simple, np.array([3,2,1]), 3)
        np.testing.assert_array_almost_equal(betas, np.array([6, -10]))


    def test_objective_with_simpledata(self):
        np.testing.assert_array_almost_equal(compute_kernelsvm_objective(np.zeros(2), x_simple, y_simple, 1), 1)

    def test_objective_with_nonzero_simpledata(self):
        np.testing.assert_array_almost_equal(compute_kernelsvm_objective(np.array([1,1]), x_simple, y_simple, 1), 2.667, decimal=3)

    def test_objective_with_nonzero_simpledata_and_diff_lambda(self):
        np.testing.assert_array_almost_equal(compute_kernelsvm_objective(np.array([1,1]), x_simple, y_simple, 3), 6.667, decimal=3)


if __name__ == '__main__':
    unittest.main()