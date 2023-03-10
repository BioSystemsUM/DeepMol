from copy import copy
from unittest import TestCase

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

    def test_run_unsupervised(self):
        self.validate_umap_classification(2)
        self.validate_umap_classification(3)
        self.validate_umap_classification(6)

        self.validate_umap_regression(2)
        self.validate_umap_regression(3)
        self.validate_umap_regression(6)

        self.validate_umap_classification(2, parametric=True)
