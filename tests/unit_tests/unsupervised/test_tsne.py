from copy import copy
from unittest import TestCase

from deepmol.unsupervised import TSNE
from unit_tests.unsupervised.test_unsupervised import UnsupervisedBaseTestCase


class TestTSNE(UnsupervisedBaseTestCase, TestCase):

    def validate_tsne_classification(self, n_components):
        dataset = copy(self.dataset)
        pca = TSNE(n_components=n_components)
        components_df = pca.run_unsupervised(dataset)
        self.assertEqual(components_df.X.shape, (dataset.X.shape[0], n_components))
        pca.plot(components_df.X, path='test_components.png')

    def validate_tsne_regression(self, n_components):
        dataset = copy(self.regression_dataset)
        pca = TSNE(n_components=n_components)
        components_df = pca.run_unsupervised(dataset)
        self.assertEqual(components_df.X.shape, (dataset.X.shape[0], n_components))
        pca.plot(components_df.X, path='test_components.png')

    def test_run_unsupervised(self):
        self.validate_tsne_classification(2)
        self.validate_tsne_classification(3)
        self.validate_tsne_classification(4)

        self.validate_tsne_regression(2)
        self.validate_tsne_regression(3)
        self.validate_tsne_regression(4)
