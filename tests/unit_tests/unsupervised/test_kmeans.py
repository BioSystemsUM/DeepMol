from copy import copy
from typing import Union
from unittest import TestCase, skip
from unittest.mock import patch

from plotly.graph_objs import Figure

from deepmol.unsupervised import KMeans
from unit_tests.unsupervised.test_unsupervised import UnsupervisedBaseTestCase


@skip("Skip KMeans tests because it takes too much time in CI")
class TestKMeans(UnsupervisedBaseTestCase, TestCase):

    def validate_kmeans(self, n_clusters: Union[str, int] = 'elbow'):
        dataset = copy(self.dataset)
        kmeans = KMeans(n_clusters=n_clusters)
        components_df = kmeans.run(dataset)
        if n_clusters == 'elbow':
            n_clusters = 1
        self.assertEqual(components_df._X.shape, (dataset.X.shape[0], n_clusters))
        kmeans.plot(components_df._X, path='test_components.png')

    @patch.object(Figure, 'show')
    def test_run_unsupervised(self, mock_show):
        self.validate_kmeans()
        self.validate_kmeans(2)
        self.validate_kmeans(3)
        self.validate_kmeans(6)
