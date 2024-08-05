from deepmol.compound_featurization.nc_mfp_generator import NC_MFP
from tests.unit_tests.featurizers.test_featurizers import FeaturizerTestCase
from unittest import TestCase

from rdkit.Chem import MolFromSmiles

class TestNC_MFP(FeaturizerTestCase, TestCase):

    def test_featurize(self):

        transformer= NC_MFP()
        bitstring = transformer._featurize(MolFromSmiles("COc1cc(ccc1O)-c1[o+]c2cc(O)cc(O)c2cc1OC1O[C@H](COC(C)=O)[C@H](O)[C@H](O)[C@H]1O"))
        self.assertEqual(bitstring.shape[0], 1)

    def test_fingerprint_with_no_ones(self):

        transformer= NC_MFP()
        bitstring = transformer._featurize(MolFromSmiles("COC1=CC(=CC(=C1OC)OC)CCN"))
        self.assertEqual(bitstring.shape[0], 1)
        
    