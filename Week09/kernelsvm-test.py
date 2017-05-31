import unittest
from kernelsvm import *

import numpy as np
from sklearn.metrics.pairwise import linear_kernel, polynomial_kernel

from sklearn.datasets import fetch_mldata
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

# simple data for some of the tests
x_simple = np.array([3,2,0,1,-1,-2]).reshape(3,2)
y_simple = np.array([1,1,-1])

# TODOs
# 1a. Figure out what's going on when I call fastgradalgo with my two functions (w/ and w/o backtracking?). Based on
# one quick check it seems like it runs - very slowly, takes noticeable time even with just 10 iterations and no
# backtracking - but doesn't converge. I can try running it in PyCharm w/ the MNIST data below, so I can step through
# and see what's happening and what's taking time.
# 1b. (if it seems like 1a is working then perhaps that shows that this is ok and doesn't need more work) After
# confirming with Corinne or others that my by-hand understanding of the gradient implementation is accurate, make
# sure that the tests I have a) have the correct values by hand and then b) that the
# 2. Need an implementation of kerneleval, in some fashion, for the polynomial kernel - however, I don't need to
# do it with kerneleval - I really just need a predict function that does the kerneleval step and the alpha weighting
# and summing.


# MNIST data to check something
# mnist = fetch_mldata('MNIST original')
# ones_and_eights = (mnist.target == 1) | (mnist.target == 8)
# X = mnist.data[ones_and_eights]
# y = mnist.target[ones_and_eights]
# X_scaled = preprocessing.scale(X)
# X_scaled_train, X_scaled_test, y_train, y_test = train_test_split(X_scaled, y)

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

    # TODO need to write/test kerneleval_polynomial? Yep - We'll need this for accuracy w/ the polynomial kernel,
    # however it could be in a separate function that just does predict and also does the weighting via alpha

    def test_gradient_with_simpledata(self):
        alphas = compute_kernelsvm_gradient(np.zeros(3), computegram_linear(x_simple), y_simple, 1)
        np.testing.assert_array_almost_equal(alphas, np.array([-14.67, -3.33, 9.33]), 2)

    def test_gradient_with_nonzero_simpledata(self):
        alphas = compute_kernelsvm_gradient(np.array([1,1,1]), computegram_linear(x_simple), y_simple, 1)
        np.testing.assert_array_almost_equal(alphas, np.array([16, 2, -8]))

    def test_gradient_with_nonzero_simpledata_with_fewer_zeros(self):
        alphas = compute_kernelsvm_gradient(np.array([1,-1,1]), computegram_linear(x_simple), np.array([3,2,1]), 1)
        np.testing.assert_array_almost_equal(alphas, np.array([4.67, -4.67, 4.67]), 2)

    def test_gradient_with_nonzero_simpledata_and_diff_lambda(self):
        alphas = compute_kernelsvm_gradient(np.array([1,-1,1]), computegram_linear(x_simple), np.array([3,2,1]), 3)
        np.testing.assert_array_almost_equal(alphas, np.array([20.67, -8.67, 4.67]), 2)


    def test_objective_with_simpledata(self):
        self.assertEqual(compute_kernelsvm_objective(np.zeros(3), computegram_linear(x_simple), y_simple, 1), 1)

    def test_objective_with_nonzero_simpledata(self):
        self.assertEqual(compute_kernelsvm_objective(np.array([1,1,1]), computegram_linear(x_simple), y_simple, 1), 5)

    def test_objective_with_nonzero_simpledata_and_diff_lambda(self):
        self.assertEqual(compute_kernelsvm_objective(np.array([1,1,1]), computegram_linear(x_simple), y_simple, 3), 15)


    # def test

if __name__ == '__main__':
    unittest.main()