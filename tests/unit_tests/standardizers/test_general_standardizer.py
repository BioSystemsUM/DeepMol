from copy import copy
from unittest import TestCase

from rdkit.Chem import rdMolDescriptors, MolFromSmiles, GetMolFrags
from rdkit.DataStructs import TanimotoSimilarity

from deepmol.standardizer import BasicStandardizer, CustomStandardizer, ChEMBLStandardizer
from unit_tests.standardizers.test_standardizers import StandardizerBaseTestCase


class GeneralStandardizer(StandardizerBaseTestCase, TestCase):

    def check_similarity(self, standardization_method: callable, **kwargs):
        not_standardized_smiles_lst = self.original_smiles
        dataset = copy(self.mock_dataset)
        # dataset = standardization_method(**kwargs).standardize(dataset) # was also tested
        standardization_method(**kwargs).standardize(dataset, inplace=True)
        for i in range(len(dataset.mols)):
            standardized_mol = dataset.mols[i]
            not_standardized_mol = MolFromSmiles(not_standardized_smiles_lst[i])
            if standardized_mol is None or not_standardized_mol is None:
                self.assertEqual(standardized_mol, not_standardized_mol)
                continue
            elif len(GetMolFrags(not_standardized_mol)) > 1:
                if standardization_method.__name__ == 'BasicStandardizer':
                    pass
                elif standardization_method.__name__ == 'CustomStandardizer':
                    tested = False
                    if '.' in not_standardized_smiles_lst[i]:
                        tested = True
                        if standardization_method(**kwargs).params['KEEP_BIGGEST']:
                            self.assertTrue('.' not in dataset._smiles[i])
                    if '+' in not_standardized_smiles_lst[i] or '-' in not_standardized_smiles_lst[i]:
                        tested = True
                        if standardization_method(**kwargs).params['NEUTRALISE_CHARGE']:
                            self.assertTrue('-' not in dataset._smiles[i])
                            self.assertTrue('+' not in dataset._smiles[i])
                    if tested:
                        continue
                    else:
                        pass
                else:
                    self.assertTrue('.' in not_standardized_smiles_lst[i])
                    self.assertTrue(len(not_standardized_smiles_lst[i]) > len(dataset._smiles[i]))
                    continue

            fp1 = rdMolDescriptors.GetMorganFingerprintAsBitVect(standardized_mol, 2)
            fp2 = rdMolDescriptors.GetMorganFingerprintAsBitVect(not_standardized_mol, 2)

            similarity = TanimotoSimilarity(fp1, fp2)
            self.assertEqual(similarity, 1)

    def test_standardize(self):
        self.check_similarity(BasicStandardizer)
        self.check_similarity(ChEMBLStandardizer)
        self.check_similarity(CustomStandardizer)

    def test_custom_strandardizer_configurations(self):
        heavy_standardisation = {
            'REMOVE_ISOTOPE': True,
            'NEUTRALISE_CHARGE': True,
            'REMOVE_STEREO': True,
            'KEEP_BIGGEST': True,
            'ADD_HYDROGEN': True,
            'KEKULIZE': True,
            'NEUTRALISE_CHARGE_LATE': True}
        self.check_similarity(CustomStandardizer, params=heavy_standardisation)
