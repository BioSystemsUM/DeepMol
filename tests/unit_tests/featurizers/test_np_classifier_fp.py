from tests.unit_tests.featurizers.test_featurizers import FeaturizerTestCase
from unittest import TestCase

from deepmol.compound_featurization import NPClassifierFP


class TestNPClassifierFingerprint(FeaturizerTestCase, TestCase):

    def test_featurize(self):
        dataset_rows_number = len(self.mock_dataset.mols)
        d = NPClassifierFP().featurize(self.mock_dataset)
        self.assertEqual(dataset_rows_number, d._X.shape[0])

    def test_feature_with_chirality(self):
        dataset_rows_number = len(self.mock_dataset.mols)
        d = NPClassifierFP(useChirality=True).featurize(self.mock_dataset)
        self.assertEqual(dataset_rows_number, d._X.shape[0])
