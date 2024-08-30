from copy import copy

from deepmol.compound_featurization import NeuralNPFP
from tests.unit_tests.featurizers.test_featurizers import FeaturizerTestCase
from unittest import TestCase

from rdkit.Chem import MolFromSmiles


class TestNeuralNPFP(FeaturizerTestCase, TestCase):

    def test_featurize_one_molecule(self):
        transformer = NeuralNPFP()
        bitstring = transformer._featurize(
            MolFromSmiles("COc1cc(ccc1O)-c1[o+]c2cc(O)cc(O)c2cc1OC1O[C@H](COC(C)=O)[C@H](O)[C@H](O)[C@H]1O"))
        self.assertEqual(bitstring.shape[0], 64)

        transformer = NeuralNPFP(model_name="base")
        bitstring = transformer._featurize(
            MolFromSmiles("COc1cc(ccc1O)-c1[o+]c2cc(O)cc(O)c2cc1OC1O[C@H](COC(C)=O)[C@H](O)[C@H](O)[C@H]1O"))
        self.assertEqual(bitstring.shape[0], 64)

        transformer = NeuralNPFP(model_name="ae")
        bitstring = transformer._featurize(
            MolFromSmiles("COc1cc(ccc1O)-c1[o+]c2cc(O)cc(O)c2cc1OC1O[C@H](COC(C)=O)[C@H](O)[C@H](O)[C@H]1O"))
        self.assertEqual(bitstring.shape[0], 64)

    def test_featurize(self):
        transformer = NeuralNPFP()
        transformer.featurize(self.mock_dataset, inplace=True)
        self.assertEqual(7, self.mock_dataset._X.shape[0])
        self.assertEqual(64, self.mock_dataset._X.shape[1])

    def test_featurize_with_nan(self):
        dataset_rows_number = len(self.mock_dataset_with_invalid.mols) - 1  # one mol has invalid smiles

        dataset = copy(self.mock_dataset_with_invalid)
        NeuralNPFP().featurize(dataset, inplace=True)
        self.assertEqual(dataset_rows_number, dataset._X.shape[0])

    def test_failed_featurize(self):

        with self.assertRaises(ValueError):
            transformer = NeuralNPFP(model_name="invalid")

    def test_fingerprint_with_no_ones(self):
        transformer = NeuralNPFP(device="cpu")
        bitstring = transformer._featurize(MolFromSmiles("COC1=CC(=CC(=C1OC)OC)CCN"))
        self.assertEqual(bitstring.shape[0], 64)
