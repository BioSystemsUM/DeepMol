from unittest import TestCase, skip

import numpy as np
from rdkit import DataStructs
from rdkit.Chem import AllChem

from deepmol.splitters import SimilaritySplitter
from unit_tests.splitters.test_splitters import TestSplitters


class TestSimilaritySplitter(TestSplitters, TestCase):

    def test_split(self):
        similarity_splitter = SimilaritySplitter()

        train_dataset, test_dataset = similarity_splitter.train_test_split(self.mini_dataset_to_test)

        self.assertGreater(len(train_dataset.smiles), len(test_dataset.smiles))
        self.assertGreater(len(train_dataset.smiles), len(test_dataset.smiles))
        self.assertEqual(len(train_dataset.smiles), 4)
        self.assertEqual(len(test_dataset.smiles), 1)

        fps1 = [AllChem.GetMorganFingerprintAsBitVect(x, 2, 1024) for x in train_dataset.mols]

        mol2 = test_dataset.mols[0]
        fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, 2, 1024)

        sim_to_compare = DataStructs.TanimotoSimilarity(fps1[0], fps1[1])
        counter = 0
        for fp in fps1:
            sim = DataStructs.TanimotoSimilarity(fp, fp2)
            if sim_to_compare > sim:
                counter += 1

        self.assertGreater(counter, len(train_dataset.smiles) / 2)

    @skip("Not implemented yet!")
    def test_k_fold_split(self):
        similarity_splitter = SimilaritySplitter()

        folds = similarity_splitter.k_fold_split(self.dataset_for_k_split, k=3, seed=123)

        for train_df, valid_df in folds:
            self.assertEqual(len(train_df.y) + len(valid_df.y), len(self.dataset_for_k_split.y))

    def test_similarity_splitter_larger_dataset(self):
        similarity_splitter = SimilaritySplitter()

        train_dataset, test_dataset = similarity_splitter.train_test_split(self.dataset_to_test)

        self.assertGreater(len(train_dataset.smiles), len(test_dataset.smiles))
        self.assertEqual(len(train_dataset.smiles), 3435)
        self.assertEqual(len(test_dataset.smiles), 859)

        fps1 = [AllChem.GetMorganFingerprintAsBitVect(x, 2, 1024) for x in train_dataset.mols]

        mol2 = test_dataset.mols[0]
        fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, 2, 1024)

        sim_to_compare = DataStructs.TanimotoSimilarity(fps1[0], fps1[1])
        counter = 0
        for fp in fps1:
            sim = DataStructs.TanimotoSimilarity(fp, fp2)
            if sim_to_compare > sim:
                counter += 1

        self.assertGreater(counter, len(train_dataset.smiles) / 2)

    def test_similarity_splitter_larger_dataset_binary_classification(self):
        similarity_splitter = SimilaritySplitter()

        train_dataset, test_dataset = similarity_splitter.train_test_split(self.binary_dataset)

        self.assertGreater(len(train_dataset.smiles), len(test_dataset.smiles))
        self.assertAlmostEqual(
            len(train_dataset.y[train_dataset.y == 1]) / len(train_dataset.y),
            len(self.binary_dataset.y[self.binary_dataset.y == 1]) / len(self.binary_dataset.y),
            delta=0.01)
        self.assertAlmostEqual(
            len(test_dataset.y[test_dataset.y == 1]) / len(test_dataset.y),
            len(self.binary_dataset.y[self.binary_dataset.y == 1]) / len(self.binary_dataset.y),
            delta=0.01)

        fps1 = [AllChem.GetMorganFingerprintAsBitVect(x, 2, 1024) for x in train_dataset.mols]

        mol2 = test_dataset.mols[0]
        fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, 2, 1024)

        sim_to_compare = DataStructs.TanimotoSimilarity(fps1[0], fps1[1])
        counter = 0
        for fp in fps1:
            sim = DataStructs.TanimotoSimilarity(fp, fp2)
            if sim_to_compare > sim:
                counter += 1

        self.assertGreater(counter, len(train_dataset.smiles) / 2)

        train_dataset, valid_dataset, test_dataset = similarity_splitter.train_valid_test_split(self.binary_dataset,
                                                                                                frac_train=0.8,
                                                                                                frac_valid=0.1)

        self.assertGreater(len(train_dataset.smiles), len(test_dataset.smiles))
        self.assertGreater(len(train_dataset.smiles), len(valid_dataset.smiles))

        self.assertAlmostEqual(
            len(train_dataset.y[train_dataset.y == 1]) / len(train_dataset.y),
            len(self.binary_dataset.y[self.binary_dataset.y == 1]) / len(self.binary_dataset.y),
            delta=0.01)
        self.assertAlmostEqual(
            len(test_dataset.y[test_dataset.y == 1]) / len(test_dataset.y),
            len(self.binary_dataset.y[self.binary_dataset.y == 1]) / len(self.binary_dataset.y),
            delta=0.01)
        self.assertAlmostEqual(
            len(valid_dataset.y[valid_dataset.y == 1]) / len(valid_dataset.y),
            len(self.binary_dataset.y[self.binary_dataset.y == 1]) / len(self.binary_dataset.y),
            delta=0.01)

    def test_similarity_splitter_invalid_smiles(self):
        similarity_splitter = SimilaritySplitter()

        train_dataset, test_dataset = similarity_splitter.train_test_split(self.invalid_smiles_dataset)
        self.assertGreater(len(train_dataset.smiles), len(test_dataset.smiles))
        self.assertEqual(len(train_dataset.smiles), 2)
        self.assertEqual(len(test_dataset.smiles), 1)

    def test_similarity_splitter_invalid_smiles_and_nan(self):
        to_add = np.zeros(4)
        self.mini_dataset_to_test.mols = np.concatenate((self.mini_dataset_to_test.mols, to_add))
        self.mini_dataset_to_test.y = np.concatenate((self.mini_dataset_to_test.y, to_add))

        similarity_splitter = SimilaritySplitter()

        train_dataset, test_dataset = similarity_splitter.train_test_split(self.invalid_smiles_dataset)
        self.assertGreater(len(train_dataset.smiles), len(test_dataset.smiles))
        self.assertEqual(len(train_dataset.smiles), 2)
        self.assertEqual(len(test_dataset.smiles), 1)
