import unittest
from linearsvm import *

import numpy as np

# simple data for tests
x_simple = np.array([3,2,0,1,-1,-2]).reshape(3,2)
y_simple = np.array([1,0,0])
beta_init_simple = np.zeros(2)

class LinearSvmFunctionsTest(unittest.TestCase):
    def test_computelinearsvmgrad_with_simpledata(self):
        betas = computelinearsvmgrad(beta_init_simple, x_simple, y_simple, 1)
        np.testing.assert_approx_equal(betas, np.array([-2, -4/3]))


class GradientImplementationTest(unittest.TestCase):
    pass


if __name__ == '__main__':
    unittest.main()