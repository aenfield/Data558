import unittest
import coorddescent

import numpy as np


class MinimizationTest(unittest.TestCase):
    def test_c_term_with_simple_data(self):
        y = np.array([2, 3, 0 - 1])
        x = np.array([4, -2, 1,
                      5, -3, 2,
                      1, 1, 3,
                      0, 1, 4]).reshape(4, 3)
        lam = 1
        beta = np.zeros(3)

        c = c_term(beta, x, y, j, lam)

        np.assert_allclose((45 / 84), c)


if __name__ == '__main__':
    unittest.main()
