import unittest
import logistic
import numpy as np

# I'm including this as a sign that in most development I do and have included tests. For this one I didn't
# originaly write tests - I was learning at the time, so it was more of a spike vs. production code - and
# I'm not going to go back and write tests now.

class LogisticRegressionFunctionsTest(unittest.TestCase):
    def test_computegrad_with_simpledata(self):
        pass


if __name__ == '__main__':
    unittest.main()