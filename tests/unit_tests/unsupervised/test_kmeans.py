from unittest import TestCase

from deepmol.unsupervised import KMeans
from unit_tests.unsupervised.test_unsupervised import UnsupervisedBaseTestCase


class TestKMeans(UnsupervisedBaseTestCase, TestCase):

    def test_run_unsupervised(self):
        km = KMeans()
        new_df = km.run_unsupervised(self.dataset)
        km.plot(new_df.X, path='test_kmeans.png')

