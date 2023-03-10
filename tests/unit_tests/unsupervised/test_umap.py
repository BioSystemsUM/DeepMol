from copy import copy
from unittest import TestCase
from unittest.mock import patch

from plotly.graph_objs import Figure

from deepmol.unsupervised import UMAP
from unit_tests.unsupervised.test_unsupervised import UnsupervisedBaseTestCase


class TestUMAP(UnsupervisedBaseTestCase, TestCase):

    def validate_umap_classification(self, n_components, **kwargs):
        dataset = copy(self.dataset)
        pca = UMAP(n_components=n_components, **kwargs)
        components_df = pca.run_unsupervised(dataset)
        self.assertEqual(components_df.X.shape, (dataset.X.shape[0], n_components))
        pca.plot(components_df.X, path='test_components.png')

    def validate_umap_regression(self, n_components, **kwargs):
        dataset = copy(self.regression_dataset)
        pca = UMAP(n_components=n_components, **kwargs)
        components_df = pca.run_unsupervised(dataset)
        self.assertEqual(components_df.X.shape, (dataset.X.shape[0], n_components))
        pca.plot(components_df.X, path='test_components.png')

    @patch.object(Figure, 'show')
    def test_run_unsupervised(self, mock_show):
        self.validate_umap_classification(2, parametric=False)
        self.validate_umap_classification(3, parametric=False)
        self.validate_umap_classification(6, parametric=False)

        self.validate_umap_regression(2, parametric=False)
        self.validate_umap_regression(3, parametric=False)
        self.validate_umap_regression(6, parametric=False)

        self.validate_umap_classification(2, parametric=True)
