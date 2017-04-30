import unittest
from pca import *

import pandas as pd
from sklearn.decomposition import PCA
from sklearn import preprocessing # for scale

# sample data for tests, and values from sklearn's PCA implementation
# to check against
np.random.seed(42)
mean_vals = [0, 0.5, 1]
d = pd.DataFrame(np.vstack([np.random.normal(mean,
                        size=(20,50)) for mean in mean_vals]))
d.insert(0, 'Class', np.repeat(['A','B','C'], 20))
d_values = d.values[:, 1:51].astype('float')
d_values_centered = preprocessing.scale(d_values, with_std=False)

pca_sklearn = PCA(50, svd_solver='randomized')
pca_sklearn.fit(d_values)


class OjaPCATest(unittest.TestCase):
    def test_just_one_eigenvector(self):
        pca_components = oja_fit(d_values_centered, 1)
        np.testing.assert_allclose(pca_components[0], pca_sklearn.components_[0])



if __name__ == '__main__':
    unittest.main()
