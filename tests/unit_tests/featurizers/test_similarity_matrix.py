from unittest import TestCase

from deepmol.compound_featurization import TanimotoSimilarityMatrix
from unit_tests.featurizers.test_featurizers import FeaturizerTestCase


class TestSimilarityMatrix(FeaturizerTestCase, TestCase):

    def test_featurize(self):
        dataset_rows_number = len(self.mock_dataset.mols)
        TanimotoSimilarityMatrix(n_molecules=dataset_rows_number).featurize(self.mock_dataset)
        self.assertEqual(dataset_rows_number, self.mock_dataset._X.shape[0])
        self.assertEqual(dataset_rows_number, self.mock_dataset._X.shape[1])
