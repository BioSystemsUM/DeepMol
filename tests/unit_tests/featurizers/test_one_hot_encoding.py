from unittest import TestCase

from deepmol.compound_featurization import SmilesOneHotEncoder
from unit_tests.featurizers.test_featurizers import FeaturizerTestCase


class TestOneHotEncoding(FeaturizerTestCase, TestCase):
    def test_featurize(self):
        dataset_rows_number = len(self.mock_dataset.mols)
        ohe = SmilesOneHotEncoder()._fit(self.mock_dataset)
        dataset = ohe._transform(self.mock_dataset)
        self.assertEqual(dataset_rows_number, dataset._X.shape[0])
