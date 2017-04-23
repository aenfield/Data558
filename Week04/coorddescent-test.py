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
    beta_iter_2 = np.array([0.536, 0, 0])

    def test_c_term_with_simple_data(self):
        c = c_term(self.beta, self.x, self.y, 1, self.lam)
        np.testing.assert_allclose(c, 46)

    def test_c_term_second_index(self):
        c = c_term(self.beta_iter_2, self.x, self.y, 2, self.lam)
        np.testing.assert_allclose(c, -4.416)

    def test_a_term_with_simple_data(self):
        a = a_term(self.x, 1)
        np.testing.assert_allclose(a, 84)

    def test_minimize_beta_term_positive_c(self):
        b_hat = minimize_beta_term(self.beta, self.x, self.y, 1, self.lam)
        np.testing.assert_allclose(b_hat, 0.536, rtol=1e-03)

    def test_minimize_beta_term_negative_c(self):
        b_hat = minimize_beta_term(self.beta_iter_2, self.x, self.y, 2, self.lam)
        np.testing.assert_allclose(b_hat, -0.1139, rtol=1e-03)

    def test_minimize_beta_term_midrange_c(self):
        # big lambda so the c term is inside it
        b_hat = minimize_beta_term(self.beta, self.x, self.y, 3, 10)
        np.testing.assert_allclose(b_hat, 0)



if __name__ == '__main__':
    unittest.main()
