import unittest
from coorddescent import *

import numpy as np

# simple data, with which to test
y = np.array([2, 3, 0, - 1])
x = np.array([4, -2, 1,
              5, -3, 2,
              1, 1, 3,
              0, 1, 4]).reshape(4, 3)
lam = 1
beta = np.zeros(3)
beta_iter_2 = np.array([0.536, 0, 0])


class CyclicCoordDescentTest(unittest.TestCase):

    # TODO step size? where/how?

    def test_one_iteration_with_one_feature(self):
        beta_updated = cycliccoorddescent(x, y, lam, max_iter=1)
        np.testing.assert_allclose(beta_updated, np.array([0.536, 0, 0], rtol=1e-3))



#        np.testing.assert_allclose(beta_updated, np.array([0.536, -0.1143, -0.1573], rtol=1e-3))



class MinimizationTest(unittest.TestCase):

    def test_c_term_with_simple_data(self):
        c = c_term(beta, x, y, 1, lam)
        np.testing.assert_allclose(c, 46)

    def test_c_term_second_index(self):
        c = c_term(beta_iter_2, x, y, 2, lam)
        np.testing.assert_allclose(c, -4.416)

    def test_a_term_with_simple_data(self):
        a = a_term(x, 1)
        np.testing.assert_allclose(a, 84)

    def test_minimize_beta_term_positive_c(self):
        b_hat = minimize_beta_term(beta, x, y, 1, lam)
        np.testing.assert_allclose(b_hat, 0.536, rtol=1e-03)

    def test_minimize_beta_term_negative_c(self):
        b_hat = minimize_beta_term(beta_iter_2, x, y, 2, lam)
        np.testing.assert_allclose(b_hat, -0.1139, rtol=1e-03)

    def test_minimize_beta_term_midrange_c(self):
        # big lambda so the c term is inside it
        b_hat = minimize_beta_term(beta, x, y, 3, 10)
        np.testing.assert_allclose(b_hat, 0)



if __name__ == '__main__':
    unittest.main()
