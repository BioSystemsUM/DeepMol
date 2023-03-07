from copy import copy
from unittest import TestCase

from deepmol.unsupervised import PCA
from unit_tests.unsupervised.test_unsupervised import UnsupervisedBaseTestCase


class TestPCA(UnsupervisedBaseTestCase, TestCase):

    def validate_pca_classification(self, n_components):
        dataset = copy(self.dataset)
        pca = PCA(n_components=n_components)
        components_df = pca.run_unsupervised(dataset)
        self.assertEqual(components_df.X.shape, (dataset.X.shape[0], n_components))
        pca.plot(components_df.X, path='test_components.png')
        pca.plot_explained_variance(path='test_explained_variance.png')

    def validate_pca_regression(self, n_components):
        dataset = copy(self.regression_dataset)
        pca = PCA(n_components=n_components)
        components_df = pca.run_unsupervised(dataset)
        self.assertEqual(components_df.X.shape, (dataset.X.shape[0], n_components))
        pca.plot(components_df.X, path='test_components.png')
        pca.plot_explained_variance(path='test_explained_variance.png')

    def test_run_unsupervised(self):
        self.validate_pca_classification(2)
        self.validate_pca_classification(3)
        self.validate_pca_classification(6)

        self.validate_pca_regression(2)
        self.validate_pca_regression(3)
        self.validate_pca_regression(6)
