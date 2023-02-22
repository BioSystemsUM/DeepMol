from copy import copy
from unittest import TestCase

from rdkit.Chem import rdMolDescriptors, MolFromSmiles
from rdkit.DataStructs import TanimotoSimilarity

from deepmol.standardizer import BasicStandardizer, CustomStandardizer, ChEMBLStandardizer
from unit_tests.standardizers.test_standardizers import StandardizerBaseTestCase


class GeneralStandardizer(StandardizerBaseTestCase, TestCase):

    def check_similarity(self, standardization_method: callable, **kwargs):
        not_standardized_smiles_lst = self.original_smiles
        dataset = copy(self.mock_dataset)
        standardization_method(**kwargs).standardize(dataset)
        for i in range(len(dataset.mols)):
            standardized_mol = MolFromSmiles(dataset.mols[i])
            not_standardized_mol = MolFromSmiles(not_standardized_smiles_lst[i])

            fp1 = rdMolDescriptors.GetMorganFingerprintAsBitVect(standardized_mol, 2)
            fp2 = rdMolDescriptors.GetMorganFingerprintAsBitVect(not_standardized_mol, 2)

            similarity = TanimotoSimilarity(fp1, fp2)
            self.assertEqual(similarity, 1)

    def test_standardize(self):
        self.check_similarity(BasicStandardizer)
        self.check_similarity(ChEMBLStandardizer)
        self.check_similarity(CustomStandardizer)
