from copy import copy
from tests.unit_tests.featurizers.test_featurizers import FeaturizerTestCase
from unittest import TestCase

from rdkit.Chem import MolFromSmiles

from deepmol.compound_featurization.biosynfoni import BiosynfoniKeys


class TestBiosynfoniKeys(FeaturizerTestCase, TestCase):

    def test_featurize(self):
        transformer = BiosynfoniKeys()
        transformer.featurize(self.mock_dataset, inplace=True)
        self.assertEqual(7, self.mock_dataset._X.shape[0])
        self.assertEqual(39, self.mock_dataset._X.shape[1])

    def test_featurize_with_nan(self):
        dataset_rows_number = len(self.mock_dataset_with_invalid.mols) - 1  # one mol has invalid smiles

        dataset = copy(self.mock_dataset_with_invalid)
        BiosynfoniKeys().featurize(dataset, inplace=True)
        self.assertEqual(dataset_rows_number, dataset._X.shape[0])

    def test_fingerprint_with_no_ones(self):
        transformer = BiosynfoniKeys()
        bitstring = transformer._featurize(MolFromSmiles("COC1=CC(=CC(=C1OC)OC)CCN"))
        self.assertEqual(bitstring.shape[0], 39)