import unittest
from midterm import *

import numpy as np
from sklearn import preprocessing # for scale


# ----
# Elastic net
# ----


# simple data, for testing
y = np.array([2, 3, 0, - 1])
x = np.array([4, -2, 1,
              5, -3, 2,
              1, 1, 3,
              0, 1, 4]).reshape(4, 3)
lam = 1
alpha = 0.9
beta = np.zeros(3)
beta_iter_2 = np.array([0.5357142857142857, 0, 0]) # from lasso, not the coef for this code but still good for testing input

class ElasticNetRegressionTest(unittest.TestCase):
    def test_obj_function_zero_beta(self):
        self.assertEqual(elasticnet_objective(beta, x, y, lam, alpha), 3.5)

    def test_obj_function_nonzero_beta(self):
        np.testing.assert_approx_equal(elasticnet_objective(np.array([2,3,1]), x, y, 2, alpha), 45.85)

    def test_obj_function_nonzero_negative_beta(self):
        np.testing.assert_approx_equal(elasticnet_objective(np.array([1,-2,3]), x, y, 2, alpha), 129.1)


class ElasticNetMinimizationTest(unittest.TestCase):
    def test_c_term_with_simple_data(self):
        c = c_term(beta, x, y, 1)
        np.testing.assert_approx_equal(c, 11.5)

    def test_c_term_second_index(self):
        c = c_term(beta_iter_2, x, y, 2)
        np.testing.assert_approx_equal(c, -1.10714, 3)

    def test_a_term_with_simple_data(self):
        a = a_term(x, 1, lam, alpha)
        np.testing.assert_approx_equal(a, 21.2)

    def test_minimize_beta_term_positive_c(self):
        b_hat = minimize_beta_term(beta, x, y, 1, lam, alpha)
        np.testing.assert_approx_equal(b_hat, 0.5, 3)

    def test_minimize_beta_term_negative_c(self):
        b_hat = minimize_beta_term(beta_iter_2, x, y, 2, lam, alpha)
        np.testing.assert_approx_equal(b_hat, -0.0269, 3)

    def test_minimize_beta_term_midrange_c(self):
        # big lambda so the c term is inside it
        b_hat = minimize_beta_term(beta, x, y, 3, 10, alpha)
        self.assertEqual(b_hat, 0)




class CoordDescentTest(unittest.TestCase):

    def test_one_iteration_with_one_feature(self):
        beta_vals = cycliccoorddescent(x, y, lam, alpha, max_iter=1)
        np.testing.assert_allclose(get_final_coefs(beta_vals), np.array([0.5, 0, 0]), rtol=1e-3)

    def test_one_iteration_with_three_features(self):
        beta_vals = cycliccoorddescent(x, y, lam, alpha, max_iter=3)
        np.testing.assert_allclose(get_final_coefs(beta_vals), np.array([0.5, -0.077922, -0.091379]), rtol=1e-3)

    def test_one_iteration_with_five_iterations(self):
        beta_vals = cycliccoorddescent(x, y, lam, alpha, max_iter=5)
        np.testing.assert_allclose(get_final_coefs(beta_vals), np.array([0.4962, -0.0893, -0.0914]), rtol=1e-3)

    def test_returns_all_visited_beta_vals(self):
        beta_vals = cycliccoorddescent(x, y, lam, alpha, max_iter=6)
        self.assertEqual(len(beta_vals), 7)

    def test_first_row_has_all_zeros(self):
        # orig code set row vals retroactively because of by reference
        beta_vals = cycliccoorddescent(x, y, lam, alpha, max_iter=2)
        np.testing.assert_allclose(beta_vals.ix[0].values, np.array([0, 0, 0]))

    def test_sequence_of_js_just_one(self):
        seq = get_sequence_of_js(3, 1)
        np.testing.assert_allclose(seq, np.array([1]))

    def test_sequence_of_js_with_three(self):
        seq = get_sequence_of_js(3, 3)
        np.testing.assert_allclose(seq, np.arange(1,4))

    def test_sequence_of_js_with_two_sets(self):
        seq = get_sequence_of_js(3, 6)
        np.testing.assert_allclose(seq, np.append(np.arange(1,4), np.arange(1,4)))

    def test_randcoorddescent(self):
        np.random.seed(42)
        beta_vals = randcoorddescent(x, y, lam, alpha, max_iter=10)
        np.testing.assert_allclose(get_final_coefs(beta_vals), np.array([0.5291, -0.04317, -0.1065]), rtol=1e-3)

    def test_noasserts_just_to_run_code(self):
        cycliccoorddescent(x, y, 10, alpha, max_iter=10)

    # TODO would be good to have a test for get_final_coefs


# ----
# PCA via Oja
# ----

# simple data for PCA (should this be in a setup method?)
num_examples = 50
num_features = 50
mean_vals = [0, 1.5, 3, 4.5, 6]
np.random.seed(42)
d = pd.DataFrame(np.vstack([np.random.normal(mean, size=(num_examples,num_features)) for mean in mean_vals]))
d.insert(0, 'Class', np.repeat(['A','B','C','D','E'], num_examples))
d_values = d.values[:, 1:num_features+1].astype('float')
d_values_centered = preprocessing.scale(d_values, with_std=False)


class CrossValidationTest(unittest.TestCase):
    def test_oja_last10_average(self):
        np.random.seed(42)
        a_0 = np.random.randn(np.size(d_values_centered, 1))  # starting point
        a_0 /= np.linalg.norm(a_0, axis=0)
        v1, lambdas = oja(copy.deepcopy(d_values_centered), a_0, 0.001, 2, 100)  # Run the algorithm for first component vector

        self.assertEqual(len(v1), 50)
        np.testing.assert_allclose(v1[:8], [-0.075981, -0.113966, -0.0583,  0.010991, -0.127164,
                                             -0.125076, 0.007268, -0.050866], rtol=1e-3)

    def test_oja_fit_runs_without_crashing(self):
        oja_fit(d_values_centered, 3, 0.001, 2, 10)



    # TODO ideally I'd pull out the cross-val code that's in the notebook and generalize it and test here



if __name__ == '__main__':
    unittest.main()
