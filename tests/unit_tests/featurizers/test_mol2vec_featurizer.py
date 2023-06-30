from copy import copy
from unittest import TestCase

from deepmol.compound_featurization.mol2vec import Mol2Vec
from tests.unit_tests.featurizers.test_featurizers import FeaturizerTestCase


class TestMol2Vec(FeaturizerTestCase, TestCase):
    def test_featurize(self):
        dataset_rows_number = len(self.mock_dataset.mols)
        Mol2Vec().featurize(self.mock_dataset, inplace=True)
        self.assertEqual(dataset_rows_number, self.mock_dataset._X.shape[0])

    def test_featurize_with_nan(self):
        dataset_rows_number = len(self.mock_dataset_with_invalid.mols) - 1  # one mol has invalid smiles

        dataset = copy(self.mock_dataset_with_invalid)
        Mol2Vec().featurize(dataset, inplace=True)
        self.assertEqual(dataset_rows_number, dataset._X.shape[0])
