import numpy as np
from rdkit.Chem import rdMolDescriptors, MolFromSmiles

from deepmol.compound_featurization import MorganFingerprint
from integration_tests.multiprocessing_operations.test_multiprocessing import TestMultiprocessing


class TestMultiprocessingFeaturizers(TestMultiprocessing):

    def test_multiprocessing_featurizers_small_dataset(self):
        MorganFingerprint().featurize(self.small_dataset_to_test, inplace=True)
        self.small_pandas_dataset = self.small_pandas_dataset.drop([7], axis=0)  # Remove the seventh row as it is
        # not a valid SMILES and DeepMol will remove it
        for j in range(len(self.small_dataset_to_test.mols)):

            mol = MolFromSmiles(self.small_pandas_dataset.iloc[j, 6])
            fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius=2)
            for i, value in enumerate(np.asarray(fp, dtype=np.float64)):
                self.assertEqual(self.small_dataset_to_test.X[j, i], value)

    def test_multiprocessing_featurizers_big_dataset(self):
        MorganFingerprint().featurize(self.big_dataset_to_test, inplace=True)
        to_remove = []
        for j in range(len(self.big_pandas_dataset.Smiles)):
            mol = MolFromSmiles(self.big_pandas_dataset.iloc[j, 6])
            if mol is None:
                to_remove.append(j)
        self.big_pandas_dataset = self.big_pandas_dataset.drop(to_remove, axis=0)
        mols = self.big_dataset_to_test.mols
        X = self.big_dataset_to_test.X
        for j in range(len(mols)):

            mol = MolFromSmiles(self.big_pandas_dataset.iloc[j, 6])
            fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius=2)
            for i, value in enumerate(np.asarray(fp, dtype=np.float64)):
                self.assertEqual(X[j, i], value)
