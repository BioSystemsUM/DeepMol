from copy import copy
from typing import Union
from unittest import TestCase

from deepmol.unsupervised import KMeans
from unit_tests.unsupervised.test_unsupervised import UnsupervisedBaseTestCase


class TestKMeans(UnsupervisedBaseTestCase, TestCase):

    def validate_kmeans(self, n_clusters: Union[str, int] = 'elbow'):
        dataset = copy(self.dataset)
        pca = KMeans(n_clusters=n_clusters)
        components_df = pca.run_unsupervised(dataset)
        if n_clusters == 'elbow':
            n_clusters = 1
        self.assertEqual(components_df.X.shape, (dataset.X.shape[0], n_clusters))
        pca.plot(components_df.X, path='test_components.png')

    def test_run_unsupervised(self):
        self.validate_kmeans()
        self.validate_kmeans(2)
        self.validate_kmeans(3)
        self.validate_kmeans(6)
