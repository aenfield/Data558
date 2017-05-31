import unittest
from kernelsvm import *

import numpy as np
from sklearn.metrics.pairwise import linear_kernel, polynomial_kernel

# simple data for some of the tests
x_simple = np.array([3,2,0,1,-1,-2]).reshape(3,2)
y_simple = np.array([1,0,0])

class KernelSvmFunctionsTest(unittest.TestCase):
    def test_computegram_linear_with_simpledata(self):
        gram = computegram_linear(x_simple)
        np.testing.assert_array_almost_equal(gram, linear_kernel(x_simple))

    def test_computegram_polynomial_with_simpledata(self):
        gram = computegram_polynomial(x_simple, 3, 1)
        np.testing.assert_array_almost_equal(gram, polynomial_kernel(x_simple, degree=3, coef0=1, gamma=1))
        # we use gamma = 1 to match what we want, overriding the function's default that normalizes based on n

    def

if __name__ == '__main__':
    unittest.main()