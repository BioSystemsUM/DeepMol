from copy import copy

from rdkit import Chem
from rdkit.Chem import MolToSmiles, MolFromSmiles

from deepmol.standardizer import BasicStandardizer, ChEMBLStandardizer, CustomStandardizer, MolecularStandardizer
from deepmol.utils import utils
from tests.integration_tests.multiprocessing_operations.test_multiprocessing import TestMultiprocessing


class TestStandardizers(TestMultiprocessing):

    def assert_order(self, molecules, dataset, method: MolecularStandardizer):
        for j in range(len(molecules)):
            m1 = Chem.MolFromSmiles(dataset.iloc[j, 6])
            if m1 is not None:
                m1 = utils.canonicalize_mol_object(m1)
                m1 = method._standardize(m1)
                self.assertEqual(MolToSmiles(m1, canonical=True), molecules[j])

    def test_standardizers_small_dataset(self):
        d1 = copy(self.small_dataset_to_test)
        invalid = [i for i in range(len(self.small_pandas_dataset.Smiles))
                   if MolFromSmiles(self.small_pandas_dataset.Smiles[i]) is None]
        pd_df_without_invalid = self.small_pandas_dataset.drop(invalid, axis=0)
        BasicStandardizer(n_jobs=2).standardize(d1, inplace=True)
        self.assert_order(d1.smiles, pd_df_without_invalid, BasicStandardizer())

        d2 = copy(self.small_dataset_to_test)
        ChEMBLStandardizer(n_jobs=2).standardize(d2, inplace=True)
        self.assert_order(d2.smiles, pd_df_without_invalid, ChEMBLStandardizer())

        d3 = copy(self.small_dataset_to_test)
        CustomStandardizer(n_jobs=2).standardize(d3, inplace=True)
        self.assert_order(d3.smiles, pd_df_without_invalid, CustomStandardizer())

    def test_standardizers_big_dataset(self):
        d1 = copy(self.big_dataset_to_test)
        invalid = [i for i in range(len(self.big_pandas_dataset.Smiles))
                   if MolFromSmiles(self.big_pandas_dataset.Smiles[i]) is None]
        pd_df_without_invalid = self.big_pandas_dataset.drop(invalid, axis=0)
        BasicStandardizer(n_jobs=10).standardize(d1, inplace=True)
        self.assert_order(d1.smiles, pd_df_without_invalid, BasicStandardizer())

        d2 = copy(self.big_dataset_to_test)
        ChEMBLStandardizer(n_jobs=10).standardize(d2, inplace=True)
        self.assert_order(d2.smiles, pd_df_without_invalid, ChEMBLStandardizer())

        d3 = copy(self.big_dataset_to_test)
        CustomStandardizer(n_jobs=10).standardize(d3, inplace=True)
        self.assert_order(d3.smiles, pd_df_without_invalid, CustomStandardizer())
