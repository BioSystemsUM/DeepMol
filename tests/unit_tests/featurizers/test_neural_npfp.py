import torch
from deepmol.compound_featurization import NeuralNPFP
from tests.unit_tests.featurizers.test_featurizers import FeaturizerTestCase
from unittest import TestCase

from rdkit.Chem import MolFromSmiles

class TestNeuralNPFP(FeaturizerTestCase, TestCase):

    def test_featurize(self):

        transformer= NeuralNPFP()
        bitstring = transformer._featurize(MolFromSmiles("COc1cc(ccc1O)-c1[o+]c2cc(O)cc(O)c2cc1OC1O[C@H](COC(C)=O)[C@H](O)[C@H](O)[C@H]1O"))
        self.assertEqual(bitstring.shape[0], 64)

        transformer= NeuralNPFP(model="base")
        bitstring = transformer._featurize(MolFromSmiles("COc1cc(ccc1O)-c1[o+]c2cc(O)cc(O)c2cc1OC1O[C@H](COC(C)=O)[C@H](O)[C@H](O)[C@H]1O"))
        self.assertEqual(bitstring.shape[0], 64)

        transformer= NeuralNPFP(model="ae")
        bitstring = transformer._featurize(MolFromSmiles("COc1cc(ccc1O)-c1[o+]c2cc(O)cc(O)c2cc1OC1O[C@H](COC(C)=O)[C@H](O)[C@H](O)[C@H]1O"))
        self.assertEqual(bitstring.shape[0], 64)

    def test_fingerprint_with_no_ones(self):
        transformer= NeuralNPFP(device="cpu")
        bitstring = transformer._featurize(MolFromSmiles("COC1=CC(=CC(=C1OC)OC)CCN"))
        self.assertEqual(bitstring.shape[0], 64)
        