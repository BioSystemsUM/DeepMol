from copy import copy
from unittest import TestCase, skip
from unittest.mock import patch

from plotly.graph_objs import Figure

from deepmol.unsupervised import PCA
from tests.unit_tests.unsupervised.test_unsupervised import UnsupervisedBaseTestCase

@skip("Skip KMeans tests because it takes too much time in CI")
class TestPCA(UnsupervisedBaseTestCase, TestCase):

    def validate_pca_classification(self, n_components):
        dataset = copy(self.dataset)
        pca = PCA(n_components=n_components)
        components_df = pca.run(dataset)
        self.assertEqual(components_df._X.shape, (dataset.X.shape[0], n_components))
        pca.plot(components_df._X, path='test_components.png')
        pca.plot_explained_variance(path='test_explained_variance.png')

    def validate_pca_regression(self, n_components):
        dataset = copy(self.regression_dataset)
        pca = PCA(n_components=n_components)
        components_df = pca.run(dataset)
        self.assertEqual(components_df._X.shape, (dataset.X.shape[0], n_components))
        pca.plot(components_df._X, path='test_components.png')
        pca.plot_explained_variance(path='test_explained_variance.png')

    @patch.object(Figure, 'show')
    def test_run_unsupervised(self, mock_show):
        self.validate_pca_classification(2)
        self.validate_pca_classification(3)
        self.validate_pca_classification(6)

        self.validate_pca_regression(2)
        self.validate_pca_regression(3)
        self.validate_pca_regression(6)
