import unittest
from coorddescent import *

import numpy as np

class MinimizationTest(unittest.TestCase):
    y = np.array([2, 3, 0, - 1])
    x = np.array([4, -2, 1,
                  5, -3, 2,
                  1, 1, 3,
                  0, 1, 4]).reshape(4, 3)
    lam = 1
    beta = np.zeros(3)

    def test_c_term_with_simple_data(self):
        c = c_term(self.beta, self.x, self.y, 1, self.lam)
        np.testing.assert_allclose(46, c)


if __name__ == '__main__':
    unittest.main()
