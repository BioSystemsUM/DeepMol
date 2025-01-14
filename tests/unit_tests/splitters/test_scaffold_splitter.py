from unittest import TestCase, skip

import numpy as np
from rdkit import DataStructs
from rdkit.Chem import AllChem

from deepmol.splitters import ScaffoldSplitter
from tests.unit_tests.splitters.test_splitters import SplittersTestCase


class ScaffoldSplitterTestCase(SplittersTestCase, TestCase):

    def test_split(self):
        scaffold_splitter = ScaffoldSplitter()

        train_dataset, test_dataset = scaffold_splitter.train_test_split(self.mini_dataset_to_test, seed=123)

        self.assertGreater(len(train_dataset.smiles), len(test_dataset.smiles))
        self.assertEqual(len(train_dataset.smiles), 5)
        self.assertEqual(len(test_dataset.smiles), 2)

    @skip("Not implemented yet!")
    def test_k_fold_split(self):
        scaffold_splitter = ScaffoldSplitter()

        folds = scaffold_splitter.k_fold_split(self.dataset_for_k_split, k=3, seed=123)

        for train_df, valid_df in folds:
            self.assertEqual(len(train_df.y) + len(valid_df.y), len(self.dataset_for_k_split.y))

    def test_scaffold_splitter_larger_dataset(self):
        scaffold_splitter = ScaffoldSplitter()

        train_dataset, test_dataset = scaffold_splitter.train_test_split(self.dataset_to_test)
        self.assertGreater(len(train_dataset.smiles), len(test_dataset.smiles))

        fps_train = [AllChem.GetMorganFingerprintAsBitVect(x, 2, 1024) for x in train_dataset.mols]
        fps_test = [AllChem.GetMorganFingerprintAsBitVect(x, 2, 1024) for x in test_dataset.mols]

        sim_train = self._calculate_mean_fingerprints_smilarity(fps_train)

        counter = 0
        for fp in fps_test:
            sim = np.mean(DataStructs.BulkTanimotoSimilarity(fp, fps_train))
            if sim > sim_train:
                counter += 1

        self.assertGreater(counter, len(test_dataset.smiles) / 2)
        self.assertEqual(len(train_dataset.smiles) + len(test_dataset.smiles), len(self.dataset_to_test.smiles))
        self.assertEqual(len(train_dataset.smiles), 3435)
        self.assertEqual(len(test_dataset.smiles), 859)

    def test_scaffold_splitter_larger_dataset_binary_classification(self):
        scaffold_splitter = ScaffoldSplitter()

        train_dataset, test_dataset = scaffold_splitter.train_test_split(self.binary_dataset,
                                                                         frac_train=0.5,
                                                                         seed=123)

        self.assertEqual(len(train_dataset.smiles), len(test_dataset.smiles))
        self.assertAlmostEqual(
            len(train_dataset.y[train_dataset.y == 1]) / len(train_dataset.y),
            len(self.binary_dataset.y[self.binary_dataset.y == 1]) / len(self.binary_dataset.y),
            delta=0.01)
        self.assertAlmostEqual(
            len(test_dataset.y[test_dataset.y == 1]) / len(test_dataset.y),
            len(self.binary_dataset.y[self.binary_dataset.y == 1]) / len(self.binary_dataset.y),
            delta=0.01)

        fps_train = [AllChem.GetMorganFingerprintAsBitVect(x, 2, 1024) for x in train_dataset.mols]
        fps_test = [AllChem.GetMorganFingerprintAsBitVect(x, 2, 1024) for x in test_dataset.mols]

        sim_train = self._calculate_mean_fingerprints_smilarity(fps_train)

        counter = 0
        for fp in fps_test:
            sim = np.mean(DataStructs.BulkTanimotoSimilarity(fp, fps_train))
            if sim > sim_train:
                counter += 1

        self.assertGreater(counter, len(test_dataset.smiles) / 2)
        self.assertEqual(len(train_dataset.smiles) + len(test_dataset.smiles), len(self.binary_dataset.smiles))

        train_dataset, valid_dataset, test_dataset = scaffold_splitter.train_valid_test_split(self.binary_dataset,
                                                                                              frac_train=0.5,
                                                                                              frac_test=0.3,
                                                                                              frac_valid=0.2)

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

    def test_scaffold_spliter_non_homogenous_dataset(self):
        scaffold_splitter = ScaffoldSplitter()

        train_dataset, test_dataset = scaffold_splitter.train_test_split(self.binary_dataset,
                                                                         frac_train=0.5,
                                                                         seed=123,
                                                                         homogenous_datasets=False)

        self.assertEqual(len(train_dataset.smiles), len(test_dataset.smiles))
        self.assertAlmostEqual(
            len(train_dataset.y[train_dataset.y == 1]) / len(train_dataset.y),
            len(self.binary_dataset.y[self.binary_dataset.y == 1]) / len(self.binary_dataset.y),
            delta=0.01)
        self.assertAlmostEqual(
            len(test_dataset.y[test_dataset.y == 1]) / len(test_dataset.y),
            len(self.binary_dataset.y[self.binary_dataset.y == 1]) / len(self.binary_dataset.y),
            delta=0.01)

        fps_train = [AllChem.GetMorganFingerprintAsBitVect(x, 2, 1024) for x in train_dataset.mols]
        fps_test = [AllChem.GetMorganFingerprintAsBitVect(x, 2, 1024) for x in test_dataset.mols]

        sim_train = self._calculate_mean_fingerprints_smilarity(fps_train)

        counter = 0
        for fp in fps_test:
            sim = np.mean(DataStructs.BulkTanimotoSimilarity(fp, fps_train))
            if sim > sim_train:
                counter += 1

        self.assertLess(counter, len(test_dataset.smiles) / 2)
        self.assertEqual(len(train_dataset.smiles) + len(test_dataset.smiles), len(self.binary_dataset.smiles))

        train_dataset, valid_dataset, test_dataset = scaffold_splitter.train_valid_test_split(self.binary_dataset,
                                                                                              frac_train=0.5,
                                                                                              frac_test=0.3,
                                                                                              frac_valid=0.2,
                                                                                              homogenous_datasets=False)

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

    def test_scaffold_splitter_invalid_smiles(self):
        scaffold_splitter = ScaffoldSplitter()

        train_dataset, test_dataset = scaffold_splitter.train_test_split(self.invalid_smiles_dataset)
        self.assertEqual(len(train_dataset.smiles), 3)
        self.assertEqual(len(test_dataset.smiles), 1)

    def test_scaffold_splitter_invalid_smiles_and_nan(self):
        to_add = np.zeros(4)
        self.mini_dataset_to_test.mols = np.concatenate((self.mini_dataset_to_test.mols, to_add))
        self.mini_dataset_to_test.y = np.concatenate((self.mini_dataset_to_test.y, to_add))

        scaffold_splitter = ScaffoldSplitter()

        train_dataset, test_dataset = scaffold_splitter.train_test_split(self.invalid_smiles_dataset)
        self.assertEqual(len(train_dataset.smiles), 3)
        self.assertEqual(len(test_dataset.smiles), 1)
