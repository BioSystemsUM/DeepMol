from copy import copy, deepcopy

from rdkit import Chem
from rdkit.Chem import MolToSmiles

from deepmol.standardizer import BasicStandardizer, ChEMBLStandardizer, CustomStandardizer, MolecularStandardizer
from deepmol.utils import utils
from integration_tests.multiprocessing_operations.test_multiprocessing import TestMultiprocessing


class TestStandardizers(TestMultiprocessing):

    def assert_order(self, molecules, dataset, method: MolecularStandardizer):
        for j in range(len(molecules)):
            m1 = Chem.MolFromSmiles(dataset.iloc[j, 6])
            m1 = utils.canonicalize_mol_object(m1)
            m1 = method._standardize(m1)
            if m1 is not None:
                self.assertEqual(MolToSmiles(m1, canonical=True), molecules[j])

    def test_standardizers_small_dataset(self):
        mols = copy(self.small_dataset_to_test.mols)
        BasicStandardizer().standardize(self.small_dataset_to_test)
        self.assert_order(self.small_dataset_to_test.mols, self.small_pandas_dataset, BasicStandardizer())

        self.small_dataset_to_test.mols = mols
        ChEMBLStandardizer().standardize(self.small_dataset_to_test)
        self.assert_order(self.small_dataset_to_test.mols, self.small_pandas_dataset, ChEMBLStandardizer())

        self.small_dataset_to_test.mols = mols
        CustomStandardizer().standardize(self.small_dataset_to_test)
        self.assert_order(self.small_dataset_to_test.mols, self.small_pandas_dataset, CustomStandardizer())

    def test_standardizers_big_dataset(self):
        mols = deepcopy(self.big_dataset_to_test.mols)
        BasicStandardizer().standardize(self.big_dataset_to_test)
        self.assert_order(self.big_dataset_to_test.mols, self.big_pandas_dataset, BasicStandardizer())

        self.big_dataset_to_test.mols = deepcopy(mols)
        ChEMBLStandardizer().standardize(self.big_dataset_to_test)
        self.assert_order(self.big_dataset_to_test.mols, self.big_pandas_dataset, ChEMBLStandardizer())

        self.big_dataset_to_test.mols = deepcopy(mols)
        CustomStandardizer().standardize(self.big_dataset_to_test)
        self.assert_order(self.big_dataset_to_test.mols, self.big_pandas_dataset, CustomStandardizer())
