from copy import copy
from unittest import TestCase

from rdkit.Chem import rdMolDescriptors, MolFromSmiles
from rdkit.DataStructs import BulkTanimotoSimilarity, TanimotoSimilarity

from standardizer.BasicStandardizer import BasicStandardizer
from standardizer.ChEMBLStandardizer import ChEMBLStandardizer
from standardizer.CustomStandardizer import CustomStandardizer
from tests.unit_tests.standardizers.test_standardizers import StandardizerBaseTestCase


class GeneralStandardizer(StandardizerBaseTestCase, TestCase):

    def check_similarity(self, standardization_method: callable, **kwargs):
        not_standardized_smiles_lst = copy(self.dataset_to_test.mols)
        dataset = copy(self.dataset_to_test)
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