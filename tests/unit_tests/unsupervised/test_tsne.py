from copy import copy
from unittest import TestCase, skip
from unittest.mock import patch

from plotly.graph_objs import Figure

from deepmol.unsupervised import TSNE
from tests.unit_tests.unsupervised.test_unsupervised import UnsupervisedBaseTestCase

@skip("Skip KMeans tests because it takes too much time in CI")
class TestTSNE(UnsupervisedBaseTestCase, TestCase):

    def validate_tsne_classification(self, n_components, **kwargs):
        dataset = copy(self.dataset)
        pca = TSNE(n_components=n_components, **kwargs)
        components_df = pca.run(dataset)
        self.assertEqual(components_df._X.shape, (dataset.X.shape[0], n_components))
        pca.plot(components_df._X, path='test_components.png')

    def validate_tsne_regression(self, n_components, **kwargs):
        dataset = copy(self.regression_dataset)
        pca = TSNE(n_components=n_components, **kwargs)
        components_df = pca.run(dataset)
        self.assertEqual(components_df._X.shape, (dataset.X.shape[0], n_components))
        pca.plot(components_df._X, path='test_components.png')

    @patch.object(Figure, 'show')
    def test_run_unsupervised(self, mock_show):
        self.validate_tsne_classification(2)
        self.validate_tsne_classification(3)
        self.validate_tsne_classification(4, method='exact')

        self.validate_tsne_regression(2)
        self.validate_tsne_regression(3)
        self.validate_tsne_regression(4, method='exact')
